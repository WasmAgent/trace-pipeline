"""Tests for compliance_to_dpo_records and cascade."""
from __future__ import annotations

from evomerge.pipeline.cascade import CascadeConfig, CascadeOutcome, CascadeRunner
from evomerge.pipeline.compliance_dpo import compliance_to_dpo_records
from evomerge.router.labels import RouterLabel
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
from evomerge.synthesize.templates import TaskType, make_task_spec

# ─── fixtures ────────────────────────────────────────────────────────────────

def _violation(cid="c1"):
    return ConstraintViolation(
        constraint_id=cid,
        level=ConstraintLevel.hard,
        category=ConstraintCategory.content,
        hint=f"Missing section for {cid}",
        detected_at=ViolationStage.post_decode,
    )

def _repair(round_n=1, cid="c1", ok=True):
    return RepairTraceEntry(
        round=round_n, violation_ids=[cid],
        strategy=RepairStrategy.insert_section, ok=ok,
    )

def _rec(task_id="t1", final_pass=True, n_violations=1, n_repair=1, fail_repair=False):
    violations = [_violation(f"c{i}") for i in range(n_violations)]
    repair = [_repair(i+1, f"c{i}", ok=(not fail_repair)) for i in range(n_repair)]
    return ComplianceEvalRecord(
        task_id=task_id,
        task_spec_hash="abc",
        model="qwen-7b",
        mode=RunMode.full_pcl,
        final_pass=final_pass,
        artifact="Final compliant output.",
        violations=violations,
        repair_trace=repair,
        repair_rounds=n_repair,
    )


# ─── compliance_to_dpo_records ───────────────────────────────────────────────

class TestComplianceToDpo:
    def test_passing_with_repair_yields_pairs(self):
        records = compliance_to_dpo_records([_rec(final_pass=True, n_repair=1)])
        assert len(records) >= 1

    def test_all_results_are_dpo_records(self):
        records = compliance_to_dpo_records([_rec()])
        assert all(isinstance(r, DpoTrainingRecord) for r in records)

    def test_failing_record_skipped(self):
        records = compliance_to_dpo_records([_rec(final_pass=False)])
        assert records == []

    def test_passing_no_repair_no_bad_outputs_skipped(self):
        rec = ComplianceEvalRecord(
            task_id="t1", task_spec_hash="abc", model="qwen-7b",
            mode=RunMode.full_pcl, final_pass=True,
            artifact="Good output.", repair_rounds=0,
        )
        records = compliance_to_dpo_records([rec])
        assert records == []

    def test_chosen_is_compliant_artifact(self):
        records = compliance_to_dpo_records([_rec()])
        for r in records:
            assert r.chosen == "Final compliant output."

    def test_rejected_differs_from_chosen(self):
        records = compliance_to_dpo_records([_rec()])
        for r in records:
            assert r.chosen != r.rejected

    def test_repair_loss_weight(self):
        records = compliance_to_dpo_records([_rec()])
        for r in records:
            assert r.loss_weight_tokens == "recovery"

    def test_provenance_source(self):
        records = compliance_to_dpo_records([_rec()])
        for r in records:
            assert r.provenance.source == "wasmagent-compliance"

    def test_multiple_repair_rounds_yield_multiple_pairs(self):
        records = compliance_to_dpo_records([_rec(n_violations=2, n_repair=2)])
        assert len(records) >= 2

    def test_failed_repair_rounds_skipped(self):
        records = compliance_to_dpo_records([_rec(n_repair=1, fail_repair=True)])
        assert records == []

    def test_bad_outputs_strategy(self):
        bad = {"t1": ["Bad output A.", "Bad output B."]}
        rec = ComplianceEvalRecord(
            task_id="t1", task_spec_hash="abc", model="qwen-7b",
            mode=RunMode.full_pcl, final_pass=True,
            artifact="Good output.", repair_rounds=0,
        )
        records = compliance_to_dpo_records([rec], bad_outputs=bad)
        assert len(records) == 2
        rejected_texts = {r.rejected for r in records}
        assert "Bad output A." in rejected_texts
        assert "Bad output B." in rejected_texts

    def test_bad_output_identical_to_artifact_skipped(self):
        rec = ComplianceEvalRecord(
            task_id="t1", task_spec_hash="abc", model="qwen-7b",
            mode=RunMode.full_pcl, final_pass=True,
            artifact="Same text.", repair_rounds=0,
        )
        records = compliance_to_dpo_records([rec], bad_outputs={"t1": ["Same text."]})
        assert records == []

    def test_schema_version(self):
        records = compliance_to_dpo_records([_rec()])
        for r in records:
            assert r.schema_version == "dpo/v1"


# ─── CascadeRunner ───────────────────────────────────────────────────────────

def _make_runner(
    small_passes: bool = True,
    repair_passes: bool = True,
    large_passes: bool = True,
    max_repair: int = 2,
    skip_router: bool = True,
) -> CascadeRunner:
    def small_fn(task):  return "small output"
    def large_fn(task):  return "large output"

    call_count = {"repair": 0}
    def repair_fn(task, artifact, hints):
        call_count["repair"] += 1
        return f"repaired output round {call_count['repair']}"

    verify_call = {"n": 0}
    def verify_fn(task, artifact):
        verify_call["n"] += 1
        if "small" in artifact:   return (small_passes, [] if small_passes else ["violation"])
        if "repaired" in artifact: return (repair_passes, [] if repair_passes else ["violation"])
        if "large" in artifact:   return (large_passes, [] if large_passes else ["violation"])
        return (False, ["unknown artifact"])

    return CascadeRunner(
        config=CascadeConfig(max_repair_rounds=max_repair, skip_router_precheck=skip_router),
        small_fn=small_fn,
        repair_fn=repair_fn,
        large_fn=large_fn,
        verify_fn=verify_fn,
    )


class TestCascadeRunner:
    def test_small_passes_tier_is_small(self):
        runner = _make_runner(small_passes=True)
        outcome = runner.run("t1", "do the task")
        assert outcome.tier_used == "small"
        assert outcome.final_pass is True
        assert outcome.escalated is False

    def test_small_fails_repair_passes(self):
        runner = _make_runner(small_passes=False, repair_passes=True)
        outcome = runner.run("t1", "do the task")
        assert outcome.tier_used == "repair"
        assert outcome.final_pass is True
        assert outcome.repair_rounds == 1

    def test_small_fails_repair_fails_large_passes(self):
        runner = _make_runner(small_passes=False, repair_passes=False, large_passes=True)
        outcome = runner.run("t1", "do the task")
        assert outcome.tier_used == "large"
        assert outcome.escalated is True
        assert outcome.final_pass is True

    def test_all_fail_tier_is_failed(self):
        runner = _make_runner(small_passes=False, repair_passes=False, large_passes=False)
        outcome = runner.run("t1", "do the task")
        assert outcome.tier_used == "failed"
        assert outcome.final_pass is False

    def test_repair_rounds_counted(self):
        runner = _make_runner(small_passes=False, repair_passes=False,
                              max_repair=3, large_passes=True)
        outcome = runner.run("t1", "task")
        assert outcome.repair_rounds == 3

    def test_outcome_has_task_id(self):
        runner = _make_runner()
        outcome = runner.run("my-task-id", "task")
        assert outcome.task_id == "my-task-id"

    def test_run_batch_returns_all(self):
        runner = _make_runner()
        tasks  = [(f"t{i}", f"task {i}") for i in range(5)]
        outcomes = runner.run_batch(tasks)
        assert len(outcomes) == 5
        assert all(isinstance(o, CascadeOutcome) for o in outcomes)

    def test_router_precheck_escalates_directly(self):
        """When router says need_large_model, small is never called."""
        small_called = {"n": 0}

        def small_fn(task):
            small_called["n"] += 1
            return "small output"

        spec = make_task_spec(TaskType.markdown_report, intent="test",
                              required_sections=["S"] * 12)  # many hard constraints → escalate

        from evomerge.router.classifier import RouterConfig
        cfg = CascadeConfig(
            skip_router_precheck=False,
            router_config=RouterConfig(hard_constraint_limit=5),
        )
        runner = CascadeRunner(
            config=cfg,
            small_fn=small_fn,
            repair_fn=lambda t, a, h: a,
            large_fn=lambda t: "large output",
            verify_fn=lambda t, a: (True, []),
            spec=spec,
        )
        outcome = runner.run("t1", "task")
        assert outcome.escalated is True
        assert small_called["n"] == 0
        assert outcome.router_label == RouterLabel.need_large_model

    def test_violation_hints_in_outcome_on_failure(self):
        runner = _make_runner(small_passes=False, repair_passes=False, large_passes=False)
        outcome = runner.run("t1", "task")
        assert len(outcome.violation_hints) > 0


# ─── top-level import check ──────────────────────────────────────────────────

class TestTopLevelImports:
    def test_router_imports_from_top_level(self):
        from evomerge import RouterLabel
        assert RouterLabel.small_model_can_handle

    def test_eval_imports_from_top_level(self):
        from evomerge import EvalHarness
        assert EvalHarness is not None

    def test_compliance_dpo_importable(self):
        from evomerge.pipeline import compliance_to_dpo_records
        assert callable(compliance_to_dpo_records)

    def test_cascade_importable(self):
        from evomerge.pipeline import CascadeRunner
        assert CascadeRunner is not None
