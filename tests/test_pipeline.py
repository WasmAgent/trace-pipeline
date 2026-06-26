"""Tests for evomerge.pipeline converters."""
from __future__ import annotations


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
from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry
from evomerge.pipeline.sft import to_sft_records
from evomerge.pipeline.dpo import to_dpo_records
from evomerge.pipeline.ppo import to_ppo_records
from evomerge.pipeline.compliance_sft import compliance_to_sft_records


def _branch(rollout_id="r1", branch_index=0, score=1, status="pass", answer="Good."):
    return RolloutBranchRecord(
        rollout_id=rollout_id,
        task="Summarise the document.",
        branch_index=branch_index,
        temperature=0.7,
        session_id="s1",
        final_answer=answer,
        objective_score=score,
        objective_status=status,
        rank=branch_index,
        total_score=float(score),
    )


def _compliance(task_id="t1", final_pass=True, n_violations=0, n_repair=0):
    violations = [
        ConstraintViolation(
            constraint_id=f"c{i}",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.format,
            hint=f"Missing section {i}",
            detected_at=ViolationStage.post_decode,
        )
        for i in range(n_violations)
    ]
    repair_trace = [
        RepairTraceEntry(
            round=i + 1,
            violation_ids=[f"c{i}"],
            strategy=RepairStrategy.insert_section,
            ok=True,
        )
        for i in range(n_repair)
    ]
    return ComplianceEvalRecord(
        task_id=task_id,
        task_spec_hash="abc",
        model="qwen-7b",
        mode=RunMode.full_pcl,
        final_pass=final_pass,
        artifact="Final text.",
        violations=violations,
        repair_trace=repair_trace,
        repair_rounds=n_repair,
    )


class TestSftPipeline:
    def test_only_passing(self):
        branches = [_branch(score=1), _branch(branch_index=1, score=0, status="fail")]
        records = to_sft_records(branches)
        assert len(records) == 1
        assert records[0].messages[-1].content == "Good."

    def test_include_all(self):
        branches = [_branch(score=1), _branch(branch_index=1, score=0, status="fail")]
        records = to_sft_records(branches, only_passing=False)
        assert len(records) == 2

    def test_message_structure(self):
        entry = ToolCallEntry(tool_name="search", arguments={"q": "test"}, result="ok")
        b = RolloutBranchRecord(
            rollout_id="r1",
            task="Do a search.",
            branch_index=0,
            temperature=0.7,
            session_id="s1",
            tool_call_sequence=[entry],
            final_answer="Result is X.",
            objective_score=1,
            objective_status="pass",
        )
        records = to_sft_records([b])
        roles = [m.role for m in records[0].messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_provenance_has_task_hash(self):
        records = to_sft_records([_branch()])
        assert records[0].provenance.task_hash is not None

    def test_empty_input(self):
        assert to_sft_records([]) == []


class TestDpoPipeline:
    """Tests for to_dpo_records basic pairing logic.

    All calls use require_attested_evidence=False because the test fixture
    branches are minimal stubs without verifier_results; attestation gate
    behaviour is tested separately in test_contamination_unicode.py.
    """

    def test_basic_pair(self):
        branches = [
            _branch(branch_index=0, score=1, status="pass", answer="Good."),
            _branch(branch_index=1, score=0, status="fail", answer="Bad."),
        ]
        records = to_dpo_records(branches, require_attested_evidence=False)
        assert len(records) == 1
        assert records[0].chosen == "Good."
        assert records[0].rejected == "Bad."

    def test_skips_unknown(self):
        branches = [
            _branch(branch_index=0, score=1, status="unknown"),
            _branch(branch_index=1, score=0, status="unknown"),
        ]
        records = to_dpo_records(branches, require_attested_evidence=False)
        assert records == []

    def test_skips_single_branch(self):
        records = to_dpo_records([_branch()], require_attested_evidence=False)
        assert records == []

    def test_different_rollouts_not_paired(self):
        branches = [
            _branch(rollout_id="r1", branch_index=0, score=1, status="pass"),
            _branch(rollout_id="r2", branch_index=0, score=0, status="fail"),
        ]
        records = to_dpo_records(branches, require_attested_evidence=False)
        assert records == []

    def test_prompt_messages_excludes_last(self):
        branches = [
            _branch(branch_index=0, score=1, status="pass"),
            _branch(branch_index=1, score=0, status="fail"),
        ]
        rec = to_dpo_records(branches, require_attested_evidence=False)[0]
        assert rec.prompt_messages is not None
        assert len(rec.prompt_messages) == len(rec.messages) - 1


class TestPpoPipeline:
    def test_all_branches_included(self):
        branches = [_branch(score=1), _branch(branch_index=1, score=0, status="fail")]
        records = to_ppo_records(branches)
        assert len(records) == 2

    def test_reward_from_score(self):
        b_pass = _branch(score=1, status="pass")
        b_fail = _branch(branch_index=1, score=0, status="fail")
        records = to_ppo_records([b_pass, b_fail])
        rewards = {r.reward for r in records}
        assert rewards == {0.0, 1.0}

    def test_custom_reward_fn(self):
        records = to_ppo_records([_branch()], reward_fn=lambda r: 0.5)
        assert records[0].reward == 0.5


class TestComplianceSftPipeline:
    def test_passing_record_yields_answerer(self):
        records = compliance_to_sft_records([_compliance(final_pass=True)])
        assert len(records) == 1
        assert records[0].output_type == "final_answer"

    def test_failing_excluded_by_default(self):
        records = compliance_to_sft_records([_compliance(final_pass=False)])
        assert records == []

    def test_failing_included_when_requested(self):
        records = compliance_to_sft_records(
            [_compliance(final_pass=False)], include_failures=True
        )
        assert len(records) == 1

    def test_repair_rounds_generate_repairer_records(self):
        rec = _compliance(final_pass=True, n_violations=2, n_repair=2)
        records = compliance_to_sft_records([rec])
        output_types = [r.output_type for r in records]
        assert "repair_patch" in output_types
        assert output_types.count("repair_patch") == 2

    def test_loss_weight_for_repair(self):
        rec = _compliance(final_pass=True, n_violations=1, n_repair=1)
        records = compliance_to_sft_records([rec])
        repair = next(r for r in records if r.output_type == "repair_patch")
        assert repair.loss_weight_tokens == "recovery"
