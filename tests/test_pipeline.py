"""Tests for evomerge.pipeline converters."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
from evomerge.schemas.training import TurnAnnotation, LossWeightTokens
from evomerge.pipeline.sft import to_sft_records, _normalize_tool_sequence, _collapse_annotations
from evomerge.pipeline.dpo import to_dpo_records
from evomerge.pipeline.ppo import to_ppo_records
from evomerge.pipeline.compliance_sft import compliance_to_sft_records
from evomerge.io import load_rollouts


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


class TestAgentEventNormalization:
    """Issue #3: AgentEvent format tool_call_sequence is normalized correctly."""

    def test_agent_event_pairs_to_tool_call_entries(self):
        agent_events = [
            {
                "channel": "tool",
                "event": "tool_call",
                "data": {
                    "toolName": "search",
                    "args": {"q": "test"},
                    "callId": "c1",
                    "batchId": "b1",
                    "batchSize": 1,
                    "stepIndex": 0,
                },
                "traceId": "t1",
                "parentTraceId": None,
                "timestampMs": 0,
            },
            {
                "channel": "tool",
                "event": "tool_result",
                "data": {
                    "callId": "c1",
                    "toolName": "search",
                    "output": "result",
                    "error": None,
                },
                "traceId": "t2",
                "parentTraceId": "t1",
                "timestampMs": 1,
            },
        ]
        entries = _normalize_tool_sequence(agent_events)
        assert len(entries) == 1
        assert entries[0].tool_name == "search"
        assert entries[0].arguments == {"q": "test"}
        assert entries[0].result == "result"
        assert entries[0].error is None

    def test_agent_event_with_error(self):
        agent_events = [
            {
                "channel": "tool",
                "event": "tool_call",
                "data": {"toolName": "fetch", "args": {"url": "x"}, "callId": "c2"},
            },
            {
                "channel": "tool",
                "event": "tool_result",
                "data": {"callId": "c2", "toolName": "fetch", "output": None, "error": "timeout"},
            },
        ]
        entries = _normalize_tool_sequence(agent_events)
        assert len(entries) == 1
        assert entries[0].error == "timeout"
        assert entries[0].result is None

    def test_passthrough_for_tool_call_entry_format(self):
        existing = [
            ToolCallEntry(tool_name="run", arguments={"cmd": "ls"}, result="ok")
        ]
        result = _normalize_tool_sequence(existing)
        assert result is existing

    def test_empty_sequence(self):
        assert _normalize_tool_sequence([]) == []

    def test_sft_pipeline_with_agent_events(self):
        """End-to-end: AgentEvent sequence in RolloutBranchRecord produces correct messages."""
        agent_events = [
            {
                "channel": "tool",
                "event": "tool_call",
                "data": {"toolName": "grep", "args": {"pattern": "foo"}, "callId": "c1"},
            },
            {
                "channel": "tool",
                "event": "tool_result",
                "data": {"callId": "c1", "toolName": "grep", "output": "line 42: foo"},
            },
        ]
        b = RolloutBranchRecord(
            rollout_id="r-agent-event",
            task="Find foo in the code.",
            branch_index=0,
            temperature=0.7,
            session_id="s1",
            tool_call_sequence=agent_events,
            final_answer="Found foo at line 42.",
            objective_score=1,
            objective_status="pass",
        )
        records = to_sft_records([b])
        assert len(records) == 1
        roles = [m.role for m in records[0].messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert records[0].messages[2].content == "line 42: foo"


class TestDpoAttestationBypass:
    """Issue #4: DPO pairing works without attestation when branches lack verifier_results."""

    def test_pairing_works_without_verifier_results(self):
        """Branches from wasmagent-rollout without verifier_results should pair."""
        branches = [
            _branch(branch_index=0, score=1, status="pass", answer="Good."),
            _branch(branch_index=1, score=0, status="fail", answer="Bad."),
        ]
        # With require_attested_evidence=True (default), should still work
        # because branches have no verifier_results at all (implicit trust)
        records = to_dpo_records(branches, require_attested_evidence=True)
        assert len(records) == 1
        assert records[0].chosen == "Good."
        assert records[0].rejected == "Bad."

    def test_branches_with_unattested_verifier_results_are_rejected(self):
        """Branches that DO have verifier_results but are NOT attested should be dropped."""
        b1 = RolloutBranchRecord.model_validate({
            "rollout_id": "r1",
            "task": "test",
            "branch_index": 0,
            "temperature": 0.7,
            "session_id": "s1",
            "final_answer": "Good.",
            "objective_score": 1,
            "objective_status": "pass",
            "verifier_results": [{"evidence_source": "self_reported"}],
        })
        b2 = RolloutBranchRecord.model_validate({
            "rollout_id": "r1",
            "task": "test",
            "branch_index": 1,
            "temperature": 0.7,
            "session_id": "s1",
            "final_answer": "Bad.",
            "objective_score": 0,
            "objective_status": "fail",
            "verifier_results": [{"evidence_source": "self_reported"}],
        })
        records = to_dpo_records([b1, b2], require_attested_evidence=True)
        assert len(records) == 0


class TestHighValueLossWeight:
    """Issue #5: 'high_value' loss_weight_tokens validates correctly."""

    def test_high_value_literal_accepted(self):
        from evomerge.schemas.training import SftTrainingRecord, Message, Provenance

        record = SftTrainingRecord(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="world"),
            ],
            output_type="final_answer",
            loss_weight_tokens="high_value",
            provenance=Provenance(source="test"),
        )
        assert record.loss_weight_tokens == "high_value"

    def test_turn_annotation_high_value(self):
        ann = TurnAnnotation(turn_index=0, loss_weight_tokens="high_value")
        assert ann.loss_weight_tokens == "high_value"

    def test_collapse_annotations_picks_highest(self):
        annotations = [
            {"turn_index": 0, "loss_weight_tokens": "default"},
            {"turn_index": 1, "loss_weight_tokens": "high_value"},
            {"turn_index": 2, "loss_weight_tokens": "recovery"},
        ]
        assert _collapse_annotations(annotations) == "high_value"

    def test_collapse_annotations_none(self):
        assert _collapse_annotations(None) == "default"

    def test_collapse_annotations_empty(self):
        assert _collapse_annotations([]) == "default"


class TestRolloutTreeRecordSkip:
    """Issue #8: RolloutTreeRecord in JSONL is skipped by load_rollouts()."""

    def test_tree_record_skipped(self):
        branch_line = json.dumps({
            "schema_version": "rollout-wire/v1",
            "rollout_id": "r1",
            "task": "Do something.",
            "branch_index": 0,
            "temperature": 0.7,
            "session_id": "s1",
            "tool_call_sequence": [],
            "final_answer": "Done.",
            "objective_score": 1,
            "objective_status": "pass",
        })
        tree_line = json.dumps({
            "schema_version": "rollout-wire/v1",
            "rollout_id": "r1",
            "task": "Do something.",
            "branches": [
                {"branch_index": 0, "forked_at_step": 3},
                {"branch_index": 1, "forked_at_step": 3},
            ],
            "fork_map": {"step-3": [0, 1]},
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(branch_line + "\n")
            f.write(tree_line + "\n")
            f.write(branch_line.replace('"branch_index": 0', '"branch_index": 1') + "\n")
            tmppath = f.name

        records = load_rollouts(tmppath)
        # Only the two branch records should be loaded, tree record skipped
        assert len(records) == 2
        assert records[0].branch_index == 0
        assert records[1].branch_index == 1
        Path(tmppath).unlink()

    def test_branch_record_with_all_fields_loads(self):
        """A branch record with seed field loads correctly."""
        line = json.dumps({
            "schema_version": "rollout-wire/v1",
            "rollout_id": "r1",
            "task": "Test seed.",
            "branch_index": 0,
            "temperature": 0.7,
            "session_id": "s1",
            "tool_call_sequence": [],
            "final_answer": "seeded.",
            "seed": 42,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(line + "\n")
            tmppath = f.name

        records = load_rollouts(tmppath)
        assert len(records) == 1
        assert records[0].seed == 42
        Path(tmppath).unlink()
