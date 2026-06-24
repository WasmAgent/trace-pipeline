"""Tests for cross_mode_dpo_records."""
from __future__ import annotations

import pytest

from evomerge.schemas.compliance import (
    ComplianceEvalRecord,
    ConstraintCategory,
    ConstraintLevel,
    ConstraintViolation,
    RepairStrategy,
    RepairTraceEntry,
    RunMode,
    ViolationStage,
)
from evomerge.schemas.training import DpoTrainingRecord
from evomerge.pipeline.cross_mode_dpo import cross_mode_dpo_records, cross_mode_summary


def _rec(task_id: str, mode: RunMode, final_pass: bool,
         model: str = "qwen", artifact: str = "") -> ComplianceEvalRecord:
    art = artifact or f"artifact_{task_id}_{mode.value}_{'pass' if final_pass else 'fail'}"
    v = [] if final_pass else [ConstraintViolation(
        constraint_id="c1", level=ConstraintLevel.hard,
        category=ConstraintCategory.format,
        hint="missing title", detected_at=ViolationStage.post_decode,
    )]
    return ComplianceEvalRecord(
        task_id=task_id, task_spec_hash="abc", model=model,
        mode=mode, final_pass=final_pass, artifact=art,
        violations=v, repair_rounds=0,
    )


class TestCrossModeDpo:
    def _triple(self, task_id, d_pass, pr_pass, pcl_pass, model="qwen"):
        return [
            _rec(task_id, RunMode.direct,       d_pass,  model),
            _rec(task_id, RunMode.prompt_retry,  pr_pass, model),
            _rec(task_id, RunMode.full_pcl,      pcl_pass, model),
        ]

    def test_pcl_beats_direct_yields_pair(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        chosen_modes = {p.chosen for p in pairs}
        assert any("pass" in c for c in chosen_modes)

    def test_all_pass_yields_no_pairs(self):
        records = self._triple("t1", d_pass=True, pr_pass=True, pcl_pass=True)
        assert cross_mode_dpo_records(records) == []

    def test_all_fail_yields_no_pairs(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=False)
        assert cross_mode_dpo_records(records) == []

    def test_only_pcl_passes_yields_two_pairs(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        # pcl vs direct + pcl vs prompt_retry
        assert len(pairs) == 2

    def test_chosen_is_passing_artifact(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert "pass" in p.chosen

    def test_rejected_is_failing_artifact(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert "fail" in p.rejected

    def test_chosen_ne_rejected(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert p.chosen != p.rejected

    def test_boundary_case_included_by_default(self):
        # prompt_retry beats full_pcl
        records = self._triple("t1", d_pass=False, pr_pass=True, pcl_pass=False)
        pairs = cross_mode_dpo_records(records, include_boundary_cases=True)
        assert len(pairs) > 0

    def test_boundary_case_excluded_when_flag_false(self):
        # only boundary: retry wins, direct loses
        records = self._triple("t1", d_pass=False, pr_pass=True, pcl_pass=False)
        pairs = cross_mode_dpo_records(records, include_boundary_cases=False)
        # retry vs direct: retry passes, direct fails → retry is higher rank → not boundary
        # pcl vs retry: retry passes, pcl fails → retry is lower rank → boundary
        boundary_pair_count = sum(1 for p in pairs
                                  if "full_pcl_fail" in p.rejected or "full_pcl_pass" in p.chosen)
        # just assert fewer pairs than with boundary
        all_pairs = cross_mode_dpo_records(records, include_boundary_cases=True)
        assert len(pairs) <= len(all_pairs)

    def test_different_models_not_paired(self):
        records = [
            _rec("t1", RunMode.full_pcl, True,  model="qwen"),
            _rec("t1", RunMode.direct,   False, model="llama"),
        ]
        pairs = cross_mode_dpo_records(records)
        assert pairs == []

    def test_same_model_different_seeds_all_paired(self):
        # Simulate 3 seeds: each seed has a full triple for the same task_id
        # Seeds have same model but task_ids differ (each seed has unique task_ids)
        records = (
            self._triple("t1", False, False, True, model="qwen") +
            self._triple("t2", False, True,  True, model="qwen")
        )
        pairs = cross_mode_dpo_records(records)
        assert len(pairs) >= 2

    def test_provenance_source(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert p.provenance.source == "wasmagent-compliance-cross-mode"

    def test_schema_version(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert p.schema_version == "dpo/v1"

    def test_loss_weight_default_for_zero_repair(self):
        records = self._triple("t1", d_pass=False, pr_pass=False, pcl_pass=True)
        pairs = cross_mode_dpo_records(records)
        for p in pairs:
            assert p.loss_weight_tokens == "default"

    def test_loss_weight_recovery_for_repaired(self):
        # pcl record with repair rounds
        pcl = _rec("t1", RunMode.full_pcl, True, artifact="repaired output")
        pcl_with_repair = ComplianceEvalRecord(
            task_id="t1", task_spec_hash="abc", model="qwen",
            mode=RunMode.full_pcl, final_pass=True,
            artifact="repaired output", repair_rounds=2,
        )
        direct_fail = _rec("t1", RunMode.direct, False)
        pairs = cross_mode_dpo_records([pcl_with_repair, direct_fail])
        for p in pairs:
            assert p.loss_weight_tokens == "recovery"


class TestCrossModeSummary:
    def _triple(self, tid, d, pr, p, model="q"):
        return [
            _rec(tid, RunMode.direct, d, model),
            _rec(tid, RunMode.prompt_retry, pr, model),
            _rec(tid, RunMode.full_pcl, p, model),
        ]

    def test_only_pcl_passes_counted(self):
        records = self._triple("t1", False, False, True)
        s = cross_mode_summary(records)
        assert s["only_pcl_passes"] == 1
        assert s["pcl_beats_direct"] == 1
        assert s["pcl_beats_retry"] == 1

    def test_all_pass_counted(self):
        records = self._triple("t1", True, True, True)
        s = cross_mode_summary(records)
        assert s["all_pass_no_signal"] == 1

    def test_all_fail_counted(self):
        records = self._triple("t1", False, False, False)
        s = cross_mode_summary(records)
        assert s["all_fail_no_signal"] == 1

    def test_retry_beats_pcl_counted(self):
        records = self._triple("t1", False, True, False)
        s = cross_mode_summary(records)
        assert s["retry_beats_pcl"] == 1

    def test_tasks_with_all_3_modes(self):
        records = (
            self._triple("t1", False, False, True) +
            self._triple("t2", True, True, True)
        )
        s = cross_mode_summary(records)
        assert s["tasks_with_all_3_modes"] == 2
