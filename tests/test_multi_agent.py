"""Tests for evomerge.graph.multi_agent."""
from __future__ import annotations

from evomerge.graph.multi_agent import (
    AgentCreditAttribution,
    MergedTraceGraph,
    SubagentNode,
    build_multi_agent_graph,
    calculate_agent_credit,
    merge_multi_agent_traces,
)
from evomerge.schemas.training import PpoTrainingRecord


def _make_record(
    run_id: str,
    trace_id: str,
    parent_trace_id: str | None = None,
    agent_id: str = "agent-1",
    passed: bool = True,
    state_changing: bool = False,
) -> dict:
    record: dict = {
        "schema_version": "aep/v0.4",
        "run_id": run_id,
        "trace_id": trace_id,
        "parent_trace_id": parent_trace_id,
        "created_at_ms": 1700000000000,
        "run_context": {"agent_id": agent_id},
        "actions": [
            {
                "action_id": f"act-{trace_id}",
                "tool_name": "bash",
                "state_changing": state_changing,
                "timestamp_ms": 1700000000001,
            }
        ],
        "verifier_results": [{"verifier_id": "v1", "passed": passed}],
    }
    if state_changing:
        record["actions"][0]["result_digest"] = "sha256:abc"
    return record


def test_build_graph_single_record():
    rec = _make_record("run-1", "trace-1")
    graph = build_multi_agent_graph([rec])
    assert len(graph.nodes) == 1
    assert len(graph.roots) == 1
    assert graph.roots[0].trace_id == "trace-1"


def test_build_graph_parent_child():
    parent = _make_record("run-1", "trace-1", agent_id="parent-agent")
    child = _make_record("run-2", "trace-2", parent_trace_id="trace-1", agent_id="child-agent")
    graph = build_multi_agent_graph([parent, child])
    assert len(graph.roots) == 1
    assert graph.roots[0].trace_id == "trace-1"
    assert len(graph.roots[0].children) == 1
    assert graph.roots[0].children[0].trace_id == "trace-2"


def test_execution_paths():
    parent = _make_record("run-1", "trace-1")
    child1 = _make_record("run-2", "trace-2", parent_trace_id="trace-1")
    child2 = _make_record("run-3", "trace-3", parent_trace_id="trace-1")
    graph = build_multi_agent_graph([parent, child1, child2])
    assert len(graph.execution_paths) == 2
    assert all(p[0] == "trace-1" for p in graph.execution_paths)


def test_calculate_credit_all_pass():
    rec = _make_record("run-1", "trace-1", state_changing=True, passed=True)
    graph = build_multi_agent_graph([rec])
    credits = calculate_agent_credit(graph)
    assert len(credits) == 1
    assert credits[0].verifier_pass_rate == 1.0
    assert credits[0].evidence_completeness == 1.0
    assert credits[0].credit_score == 1.0


def test_calculate_credit_failed_verifier():
    rec = _make_record("run-1", "trace-1", passed=False)
    graph = build_multi_agent_graph([rec])
    credits = calculate_agent_credit(graph)
    assert credits[0].verifier_pass_rate == 0.0
    assert credits[0].credit_score < 1.0


def test_merge_multi_agent_traces_empty():
    result = merge_multi_agent_traces([])
    assert result == []


def test_merge_multi_agent_traces_single():
    rec = _make_record("run-1", "trace-1", agent_id="my-agent", passed=True)
    records = merge_multi_agent_traces([rec])
    assert len(records) == 1
    assert isinstance(records[0], PpoTrainingRecord)
    assert 0.0 <= records[0].reward <= 1.0


def test_merge_multi_agent_traces_parent_child():
    parent = _make_record("run-1", "trace-1", agent_id="root-agent")
    child = _make_record("run-2", "trace-2", parent_trace_id="trace-1", agent_id="sub-agent", passed=True)
    records = merge_multi_agent_traces([parent, child])
    assert len(records) == 1
    ppo = records[0]
    assert len(ppo.messages) >= 3
