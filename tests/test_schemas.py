"""Tests for evomerge.schemas — round-trip serialisation."""
from __future__ import annotations

from evomerge.schemas.compliance import (
    ComplianceEvalRecord,
    ConstraintCategory,
    ConstraintLevel,
    ConstraintViolation,
    RepairStrategy,
    RepairTraceEntry,
    RunMode,
    TaskSpec,
    ViolationStage,
)
from evomerge.schemas.rollout import BuildResult, RolloutBranchRecord, ToolCallEntry
from evomerge.schemas.training import (
    DpoTrainingRecord,
    Message,
    PpoTrainingRecord,
    Provenance,
    SftTrainingRecord,
)


def _rollout(**kw) -> RolloutBranchRecord:
    defaults = dict(
        rollout_id="r1",
        task="write a report",
        branch_index=0,
        temperature=0.7,
        session_id="s1",
        final_answer="The answer is 42.",
        objective_score=1,
        objective_status="pass",
        rank=1,
        total_score=1.0,
    )
    return RolloutBranchRecord(**(defaults | kw))


def _compliance_record(**kw) -> ComplianceEvalRecord:
    defaults = dict(
        task_id="t1",
        task_spec_hash="abc123",
        model="qwen-7b",
        mode=RunMode.full_pcl,
        final_pass=True,
        artifact="Final answer text.",
        latency_ms=120.0,
    )
    return ComplianceEvalRecord(**(defaults | kw))


class TestRolloutSchema:
    def test_round_trip(self):
        r = _rollout()
        raw = r.model_dump_json()
        r2 = RolloutBranchRecord.model_validate_json(raw)
        assert r2.rollout_id == r.rollout_id
        assert r2.schema_version == "rollout-wire/v1"

    def test_tool_call_sequence(self):
        entry = ToolCallEntry(
            tool_name="web_search",
            arguments={"query": "foo"},
            result={"hits": []},
        )
        r = _rollout(tool_call_sequence=[entry])
        assert r.tool_call_sequence[0].tool_name == "web_search"

    def test_build_result(self):
        r = _rollout(build_result=BuildResult(status="fail", exit_code=1))
        raw = r.model_dump_json()
        r2 = RolloutBranchRecord.model_validate_json(raw)
        assert r2.build_result.status == "fail"

    def test_defaults(self):
        r = _rollout()
        assert r.objective_status == "pass"
        assert r.tool_call_sequence == []


class TestComplianceSchema:
    def test_task_spec_round_trip(self):
        spec = TaskSpec(id="ts1", intent="Write a summary", language="zh-CN")
        raw = spec.model_dump_json()
        spec2 = TaskSpec.model_validate_json(raw)
        assert spec2.id == "ts1"
        assert spec2.language == "zh-CN"

    def test_compliance_eval_record_pass(self):
        rec = _compliance_record()
        assert rec.final_pass is True
        assert rec.repair_rounds == 0

    def test_compliance_eval_record_with_violation(self):
        v = ConstraintViolation(
            constraint_id="c1",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.format,
            hint="Missing section: Summary",
            detected_at=ViolationStage.post_decode,
        )
        entry = RepairTraceEntry(
            round=1,
            violation_ids=["c1"],
            strategy=RepairStrategy.insert_section,
            ok=True,
        )
        rec = _compliance_record(
            violations=[v], repair_trace=[entry], repair_rounds=1
        )
        assert rec.violations[0].constraint_id == "c1"
        assert rec.repair_trace[0].ok is True

    def test_round_trip(self):
        rec = _compliance_record()
        raw = rec.model_dump_json()
        rec2 = ComplianceEvalRecord.model_validate_json(raw)
        assert rec2.task_id == "t1"


class TestTrainingSchema:
    def _prov(self):
        return Provenance(source="test", rollout_id="r1")

    def test_sft_round_trip(self):
        rec = SftTrainingRecord(
            messages=[
                Message(role="user", content="Do the task."),
                Message(role="assistant", content="Done."),
            ],
            output_type="final_answer",
            provenance=self._prov(),
        )
        raw = rec.model_dump_json()
        rec2 = SftTrainingRecord.model_validate_json(raw)
        assert rec2.schema_version == "sft/v1"
        assert rec2.messages[1].content == "Done."

    def test_dpo_round_trip(self):
        msgs = [
            Message(role="user", content="Task."),
            Message(role="assistant", content="Good answer."),
        ]
        rec = DpoTrainingRecord(
            messages=msgs,
            chosen="Good answer.",
            rejected="Bad answer.",
            provenance=self._prov(),
        )
        raw = rec.model_dump_json()
        rec2 = DpoTrainingRecord.model_validate_json(raw)
        assert rec2.schema_version == "dpo/v1"
        assert rec2.rejected == "Bad answer."

    def test_ppo_round_trip(self):
        rec = PpoTrainingRecord(
            messages=[
                Message(role="user", content="Task."),
                Message(role="assistant", content="Answer."),
            ],
            reward=0.8,
            provenance=self._prov(),
        )
        raw = rec.model_dump_json()
        rec2 = PpoTrainingRecord.model_validate_json(raw)
        assert rec2.reward == 0.8
