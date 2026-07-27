"""Tests for evomerge.eval (metrics + harness)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evomerge.eval.harness import EvalConfig, EvalGroup, EvalHarness
from evomerge.eval.metrics import EvalRecord, compute_metrics


def _rec(group="A", final_pass=True, tool_calls=2, valid_calls=2,
         repair_rounds=0, repair_ok=0, evidence=True, escalated=False,
         prompt_tok=100, gen_tok=200, repair_tok=0, latency=500.0,
         task_id="t1"):
    return EvalRecord(
        task_id=task_id,
        group=group,
        final_pass=final_pass,
        tool_calls_total=tool_calls,
        tool_calls_valid=valid_calls,
        repair_rounds=repair_rounds,
        repair_rounds_ok=repair_ok,
        has_evidence=evidence,
        escalated=escalated,
        prompt_tokens=prompt_tok,
        generation_tokens=gen_tok,
        repair_tokens=repair_tok,
        latency_ms=latency,
    )


class TestComputeMetrics:
    def test_all_passing(self):
        records = [_rec(group="C", final_pass=True) for _ in range(10)]
        m = compute_metrics(records)
        assert m.taskspec_pass_rate == 1.0
        assert m.n == 10

    def test_mixed_pass(self):
        records = [_rec(final_pass=(i % 2 == 0)) for i in range(10)]
        m = compute_metrics(records)
        assert abs(m.taskspec_pass_rate - 0.5) < 0.01

    def test_tool_call_validity(self):
        records = [_rec(tool_calls=4, valid_calls=3)]
        m = compute_metrics(records)
        assert abs(m.tool_call_validity - 0.75) < 0.01

    def test_no_tool_calls_validity_is_one(self):
        records = [_rec(tool_calls=0, valid_calls=0)]
        m = compute_metrics(records)
        assert m.tool_call_validity == 1.0

    def test_repair_success_rate(self):
        records = [_rec(repair_rounds=4, repair_ok=3)]
        m = compute_metrics(records)
        assert abs(m.repair_success_rate - 0.75) < 0.01

    def test_fallback_rate(self):
        records = [_rec(escalated=True)] * 3 + [_rec(escalated=False)] * 7
        m = compute_metrics(records)
        assert abs(m.fallback_rate - 0.3) < 0.01

    def test_cost_only_for_accepted(self):
        passing = _rec(final_pass=True, prompt_tok=100, gen_tok=200)
        failing = _rec(task_id="t2", final_pass=False, prompt_tok=1000, gen_tok=2000)
        m = compute_metrics([passing, failing])
        assert m.cost_per_accepted_task == 300.0

    def test_latency_only_for_accepted(self):
        passing = _rec(final_pass=True, latency=400.0)
        failing = _rec(task_id="t2", final_pass=False, latency=9999.0)
        m = compute_metrics([passing, failing])
        assert m.latency_per_accepted_ms == 400.0

    def test_wilson_ci_reasonable(self):
        records = [_rec(final_pass=True)] * 8 + [_rec(task_id=f"f{i}", final_pass=False) for i in range(2)]
        m = compute_metrics(records)
        lo, hi = m.taskspec_pass_ci
        assert lo < 0.8 < hi
        assert 0.0 <= lo <= hi <= 1.0

    def test_general_score_aggregated(self):
        records = [_rec() for _ in range(3)]
        records[0].general_score = 0.9
        records[1].general_score = 0.8
        records[2].general_score = 0.7
        m = compute_metrics(records)
        assert abs(m.general_ability_score - 0.8) < 0.01

    def test_no_general_score_is_none(self):
        m = compute_metrics([_rec()])
        assert m.general_ability_score is None

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_to_dict_keys(self):
        m = compute_metrics([_rec()])
        d = m.to_dict()
        assert "taskspec_pass_rate" in d
        assert "tool_call_validity" in d
        assert "cost_per_accepted_task" in d


class TestEvalHarness:
    def _make_harness(self, groups=("A", "C"), n_tasks=4):
        def make_run_fn(label):
            def run_fn(task_id, task):
                return _rec(
                    group=label,
                    task_id=task_id,
                    final_pass=(label == "C"),  # C always passes, A never does
                )
            return run_fn

        task_ids = [f"t{i}" for i in range(n_tasks)]
        tasks = [f"Task {i}" for i in range(n_tasks)]
        cfg = EvalConfig(task_ids=task_ids, tasks=tasks)
        grps = {g: EvalGroup(label=g, run_fn=make_run_fn(g)) for g in groups}
        return EvalHarness(config=cfg, groups=grps)

    def test_report_has_all_groups(self):
        harness = self._make_harness(groups=["A", "C"])
        report = harness.run()
        assert "A" in report.metrics
        assert "C" in report.metrics

    def test_c_outperforms_a(self):
        harness = self._make_harness()
        report = harness.run()
        assert report.metrics["C"].taskspec_pass_rate > report.metrics["A"].taskspec_pass_rate

    def test_n_matches_tasks(self):
        harness = self._make_harness(n_tasks=6)
        report = harness.run()
        for m in report.metrics.values():
            assert m.n == 6

    def test_errors_collected_when_no_stop(self):
        def failing_fn(task_id, task):
            raise RuntimeError("model error")

        cfg = EvalConfig(task_ids=["t1"], tasks=["task"])
        grps = {"A": EvalGroup(label="A", run_fn=failing_fn)}
        harness = EvalHarness(config=cfg, groups=grps)
        report = harness.run()
        assert len(report.errors["A"]) == 1

    def test_stop_on_error_raises(self):
        def failing_fn(task_id, task):
            raise RuntimeError("boom")

        cfg = EvalConfig(task_ids=["t1"], tasks=["task"], stop_on_error=True)
        grps = {"A": EvalGroup(label="A", run_fn=failing_fn)}
        harness = EvalHarness(config=cfg, groups=grps)
        with pytest.raises(RuntimeError):
            harness.run()

    def test_mismatched_task_ids_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(task_ids=["a", "b"], tasks=["x"])

    def test_save_creates_json(self):
        harness = self._make_harness()
        report = harness.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            harness.save(report, path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "metrics" in data

    def test_summary_table_has_one_row_per_group(self):
        harness = self._make_harness(groups=["A", "B", "C"])
        report = harness.run()
        table = report.summary_table()
        assert len(table) == 3
