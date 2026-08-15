"""Multi-agent trace graph merging and PPO training record generation.

Stitches parent-child AEP trace records into a unified execution graph,
computes per-agent credit attribution, and emits PpoTrainingRecord objects
for PPO/GRPO advantage training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from evomerge.schemas.training import Message, PpoTrainingRecord, Provenance, TurnAnnotation


@dataclass
class SubagentNode:
    """One AEP trace record represented as a graph node."""
    trace_id: str
    parent_trace_id: str | None
    agent_id: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    children: list["SubagentNode"] = field(default_factory=list)
    run_id: str = ""


@dataclass
class MergedTraceGraph:
    """Merged execution graph for a multi-agent trace session."""
    roots: list[SubagentNode] = field(default_factory=list)
    nodes: dict[str, SubagentNode] = field(default_factory=dict)
    execution_paths: list[list[str]] = field(default_factory=list)


@dataclass
class AgentCreditAttribution:
    """Per-agent credit/advantage score derived from the merged graph."""
    agent_id: str
    trace_id: str
    verifier_pass_rate: float
    evidence_completeness: float
    credit_score: float


def _agent_id_from_record(record: dict[str, Any]) -> str:
    ctx = record.get("run_context") or {}
    return ctx.get("agent_id") or ctx.get("subagent_id") or record.get("run_id", "unknown")


def build_multi_agent_graph(records: list[dict[str, Any]]) -> MergedTraceGraph:
    """Build a MergedTraceGraph from a list of AEP trace records.

    Records may be aep/v0.3 or aep/v0.4. Links child traces to parents
    using parent_trace_id. Records without a parent become roots.
    """
    graph = MergedTraceGraph()

    for rec in records:
        tid = rec.get("trace_id") or rec.get("run_id", "")
        node = SubagentNode(
            trace_id=tid,
            parent_trace_id=rec.get("parent_trace_id"),
            agent_id=_agent_id_from_record(rec),
            actions=rec.get("actions", []),
            verifier_results=rec.get("verifier_results", []),
            run_id=rec.get("run_id", ""),
        )
        graph.nodes[tid] = node

    for node in graph.nodes.values():
        if node.parent_trace_id and node.parent_trace_id in graph.nodes:
            graph.nodes[node.parent_trace_id].children.append(node)
        else:
            graph.roots.append(node)

    def _dfs(n: SubagentNode, path: list[str]) -> None:
        path = path + [n.trace_id]
        if not n.children:
            graph.execution_paths.append(path)
        for child in n.children:
            _dfs(child, path)

    for root in graph.roots:
        _dfs(root, [])

    return graph


def calculate_agent_credit(graph: MergedTraceGraph) -> list[AgentCreditAttribution]:
    """Compute per-agent credit attribution from a MergedTraceGraph."""
    attributions: list[AgentCreditAttribution] = []
    for node in graph.nodes.values():
        vr = node.verifier_results
        pass_rate = (
            sum(1 for v in vr if v.get("passed", False)) / len(vr) if vr else 1.0
        )
        sc_actions = [a for a in node.actions if a.get("state_changing")]
        sc_with_evidence = [
            a for a in sc_actions if a.get("result_digest") or a.get("evidence_refs")
        ]
        ev_completeness = (
            len(sc_with_evidence) / len(sc_actions) if sc_actions else 1.0
        )
        credit = 0.6 * pass_rate + 0.4 * ev_completeness
        attributions.append(AgentCreditAttribution(
            agent_id=node.agent_id,
            trace_id=node.trace_id,
            verifier_pass_rate=pass_rate,
            evidence_completeness=ev_completeness,
            credit_score=credit,
        ))
    return attributions


def _node_to_message(node: SubagentNode) -> Message:
    if not node.actions:
        return Message(role="assistant", content=f"[agent={node.agent_id}] No actions taken.")
    summaries = [
        f"{a.get('tool_name', '?')}(id={a.get('action_id', '?')})"
        for a in node.actions[:5]
    ]
    more = f" +{len(node.actions)-5} more" if len(node.actions) > 5 else ""
    return Message(
        role="assistant",
        content=f"[agent={node.agent_id}] Actions: {', '.join(summaries)}{more}",
    )


def _task_hash(node: SubagentNode) -> str:
    payload = json.dumps({"trace_id": node.trace_id, "run_id": node.run_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def merge_multi_agent_traces(records: list[dict[str, Any]]) -> list[PpoTrainingRecord]:
    """Merge AEP records into PpoTrainingRecord objects in causal order."""
    if not records:
        return []

    graph = build_multi_agent_graph(records)
    credits = {a.trace_id: a for a in calculate_agent_credit(graph)}
    result: list[PpoTrainingRecord] = []

    system_msg = Message(
        role="system",
        content="You are a multi-agent system. Trace records are stitched in causal order.",
    )

    for path in graph.execution_paths:
        messages: list[Message] = [system_msg]
        annotations: list[TurnAnnotation] = []

        for i, tid in enumerate(path):
            node = graph.nodes.get(tid)
            if node is None:
                continue
            messages.append(_node_to_message(node))
            weight = "high_value" if i == len(path) - 1 else "default"
            annotations.append(TurnAnnotation(turn_index=i + 1, loss_weight_tokens=weight))

        leaf_tid = path[-1] if path else ""
        leaf_credit = credits.get(leaf_tid)
        reward = leaf_credit.credit_score if leaf_credit else 0.5
        leaf_node = graph.nodes.get(leaf_tid)
        prov = Provenance(
            source="multi-agent-graph",
            rollout_id=leaf_node.run_id if leaf_node else "",
            task_hash=_task_hash(leaf_node) if leaf_node else "",
        )
        result.append(PpoTrainingRecord(
            messages=messages,
            reward=reward,
            loss_weight_tokens="default",
            loss_weight_annotations=annotations or None,
            provenance=prov,
        ))

    return result


__all__ = [
    "AgentCreditAttribution",
    "MergedTraceGraph",
    "SubagentNode",
    "build_multi_agent_graph",
    "calculate_agent_credit",
    "merge_multi_agent_traces",
]
