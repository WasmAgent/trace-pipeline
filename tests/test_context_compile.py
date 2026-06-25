"""Tests for evomerge.context_compile.compiler."""
from __future__ import annotations

from evomerge.context_compile.compiler import (
    ContextQaRecord,
    RouterCriticRecord,
    compile_trace,
)
from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry


def _make(
    rollout_id: str = "r1",
    branch_index: int = 0,
    n_tools: int = 3,
    status: str = "pass",
) -> RolloutBranchRecord:
    return RolloutBranchRecord(
        rollout_id=rollout_id,
        task="Fix the KV read error",
        branch_index=branch_index,
        temperature=0.7,
        session_id="sess-001",
        tool_call_sequence=[
            ToolCallEntry(
                tool_name=f"tool_{i}",
                arguments={"path": f"/workspace/file_{i}.ts"},
                result=f"content_{i}",
            )
            for i in range(n_tools)
        ],
        final_answer="Fixed by adding null check.",
        objective_score=1 if status == "pass" else 0,
        objective_status=status,
        total_score=10.0 if status == "pass" else 0.0,
    )


# ── long_context_qa ───────────────────────────────────────────────────────────

class TestLongContextQa:
    def test_passing_branch_produces_one_record(self):
        result = compile_trace(_make(status="pass"), mode="long_context_qa")
        assert len(result) == 1
        assert isinstance(result[0], ContextQaRecord)

    def test_failing_branch_produces_no_record(self):
        result = compile_trace(_make(status="fail"), mode="long_context_qa")
        assert len(result) == 0

    def test_record_fields(self):
        rec = compile_trace(_make(), mode="long_context_qa")[0]
        assert isinstance(rec, ContextQaRecord)
        assert rec.schema_version == "context-qa/v1"
        assert rec.question == "Fix the KV read error"
        assert rec.answer == "Fixed by adding null check."
        assert rec.n_tool_calls == 3
        assert rec.objective_status == "pass"

    def test_context_contains_task_and_tools(self):
        rec = compile_trace(_make(n_tools=2), mode="long_context_qa")[0]
        assert "Task:" in rec.context
        assert "tool_0" in rec.context
        assert "tool_1" in rec.context
        assert "Final answer:" in rec.context

    def test_context_truncates_long_results(self):
        long_result = "x" * 1000
        record = _make(n_tools=1)
        record.tool_call_sequence[0].result = long_result
        rec = compile_trace(record, mode="long_context_qa")[0]
        # Result is truncated to 512 chars in serialisation
        assert len(rec.context) < 5000

    def test_min_tool_calls_filters_short_traces(self):
        result = compile_trace(_make(n_tools=1), mode="long_context_qa", min_tool_calls=2)
        assert len(result) == 0

    def test_record_id_is_deterministic(self):
        r = _make()
        r1 = compile_trace(r, mode="long_context_qa")[0]
        r2 = compile_trace(r, mode="long_context_qa")[0]
        assert r1.record_id == r2.record_id

    def test_provenance_fields(self):
        rec = compile_trace(_make(rollout_id="test-001", branch_index=2), mode="long_context_qa")[0]
        assert rec.provenance["rollout_id"] == "test-001"
        assert rec.provenance["branch_index"] == 2


# ── router_critic ─────────────────────────────────────────────────────────────

class TestRouterCritic:
    def test_produces_one_record_per_step(self):
        result = compile_trace(_make(n_tools=3), mode="router_critic")
        assert len(result) == 3

    def test_all_records_are_router_critic_type(self):
        result = compile_trace(_make(n_tools=2), mode="router_critic")
        for rec in result:
            assert isinstance(rec, RouterCriticRecord)

    def test_last_step_passing_is_stop(self):
        result = compile_trace(_make(n_tools=3, status="pass"), mode="router_critic")
        assert result[-1].decision == "stop"

    def test_last_step_failing_is_ask(self):
        result = compile_trace(_make(n_tools=3, status="fail"), mode="router_critic")
        assert result[-1].decision == "ask"

    def test_intermediate_steps_are_continue(self):
        result = compile_trace(_make(n_tools=4, status="pass"), mode="router_critic")
        for rec in result[:-1]:
            assert rec.decision == "continue"

    def test_step_indices_sequential(self):
        result = compile_trace(_make(n_tools=3), mode="router_critic")
        for i, rec in enumerate(result):
            assert rec.step_index == i

    def test_partial_context_grows_with_steps(self):
        result = compile_trace(_make(n_tools=3), mode="router_critic")
        lengths = [len(r.partial_context) for r in result]
        assert lengths == sorted(lengths)

    def test_n_remaining_steps_decreases(self):
        result = compile_trace(_make(n_tools=4), mode="router_critic")
        remaining = [r.n_remaining_steps for r in result]
        assert remaining == [3, 2, 1, 0]

    def test_failing_branch_also_produces_records(self):
        result = compile_trace(_make(n_tools=2, status="fail"), mode="router_critic")
        assert len(result) == 2

    def test_single_tool_call_produces_one_record(self):
        result = compile_trace(_make(n_tools=1), mode="router_critic")
        assert len(result) == 1
        # Single step = last step
        assert result[0].decision in ("stop", "ask")

    def test_zero_tools_skipped_by_min_tool_calls(self):
        result = compile_trace(_make(n_tools=0), mode="router_critic", min_tool_calls=1)
        assert len(result) == 0
