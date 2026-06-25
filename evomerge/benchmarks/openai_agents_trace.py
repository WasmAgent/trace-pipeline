"""Cross-framework trace import: OpenAI Agents SDK trace to AEP record.

Reference: https://openai.github.io/openai-agents-python/tracing/

OpenAI Agents SDK emits spans with type: agent.run, llm.generate, tool.call,
handoff, guardrail, custom. This module converts a full trace to an AEP record.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OAISpan:
    trace_id: str
    span_id: str
    type: str
    name: str
    started_at: str
    ended_at: str
    parent_span_id: str | None = None
    attributes: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class OAITrace:
    trace_id: str
    model_id: str
    spans: list
    session_id: str = ""
    agent_name: str = ""


def _iso_to_ms(iso: str) -> float:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except (ValueError, OSError):
        return 0.0


def oai_trace_to_aep(trace: OAITrace) -> dict:
    actions = []
    verifier_results = []
    capability_decisions = []

    for span in trace.spans:
        if span.type == "tool.call":
            tool_name = span.attributes.get("tool_name", span.name)
            actions.append({
                "action_id": span.span_id,
                "tool_name": tool_name,
                "state_changing": span.attributes.get("state_changing", False),
                "precondition_digest": None,
                "result_digest": None,
                "evidence_refs": [f"oai-span/{span.span_id}"],
                "timestamp_ms": _iso_to_ms(span.started_at),
            })
            if span.error:
                capability_decisions.append({
                    "capability": f"tool:{tool_name}",
                    "subject": trace.agent_name or "agent",
                    "resource": tool_name,
                    "decision": "deny",
                    "reason_code": "tool_error",
                })
        elif span.type == "guardrail":
            passed = span.error is None and span.attributes.get("triggered", False) is False
            verifier_results.append({
                "verifier_id": f"guardrail/{span.name}",
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "claim_ids": [span.span_id],
            })

    root_spans = [s for s in trace.spans if s.type == "agent.run" and s.parent_span_id is None]
    created_at_ms = _iso_to_ms(root_spans[0].started_at) if root_spans else 0.0

    return {
        "schema_version": "aep/v0.1",
        "run_id": f"oai-agents/{trace.trace_id}",
        "trace_id": trace.trace_id,
        "model_id": trace.model_id,
        "model_provider": "openai",
        "input_refs": [],
        "output_refs": [{"uri": f"oai-agents/session/{trace.session_id}"}],
        "capability_decisions": capability_decisions,
        "actions": actions,
        "verifier_results": verifier_results,
        "created_at_ms": created_at_ms,
    }


def load_oai_trace_jsonl(path: str) -> list:
    import json

    raw_spans = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_spans.append(json.loads(line))

    by_trace = {}
    for s in raw_spans:
        tid = s.get("trace_id", "unknown")
        by_trace.setdefault(tid, []).append(s)

    traces = []
    for trace_id, spans in by_trace.items():
        oai_spans = [
            OAISpan(
                trace_id=trace_id,
                span_id=s.get("span_id", ""),
                type=s.get("type", "custom"),
                name=s.get("name", ""),
                started_at=s.get("started_at", ""),
                ended_at=s.get("ended_at", ""),
                parent_span_id=s.get("parent_span_id"),
                attributes=s.get("attributes", {}),
                error=s.get("error"),
            )
            for s in spans
        ]
        llm_spans = [s for s in oai_spans if s.type == "llm.generate"]
        model_id = (
            llm_spans[0].attributes.get("model", "unknown") if llm_spans else "unknown"
        )
        traces.append(OAITrace(
            trace_id=trace_id,
            model_id=model_id,
            spans=oai_spans,
            session_id=spans[0].get("session_id", ""),
            agent_name=spans[0].get("agent_name", ""),
        ))
    return traces
