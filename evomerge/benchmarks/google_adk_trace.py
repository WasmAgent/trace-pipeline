"""Google ADK (Agent Development Kit) trace import.
Reference: https://google.github.io/adk-docs/
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ADKEvent:
    event_id: str
    type: str  # model_response|function_call|function_response|agent_transfer|final_response
    timestamp: str  # ISO-8601
    agent_name: str = ""
    content: dict = field(default_factory=dict)
    error: str | None = None

@dataclass
class ADKTrace:
    trace_id: str
    model_id: str
    events: list  # list[ADKEvent]
    session_id: str = ""
    root_agent: str = ""

def _iso_to_ms(iso):
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except Exception:
        return 0.0

def adk_trace_to_aep(trace: ADKTrace) -> dict:
    actions = []
    verifiers = []
    for ev in trace.events:
        if ev.type == "function_call":
            fn = ev.content.get("function_name", ev.event_id)
            actions.append({"action_id": ev.event_id, "tool_name": fn, "state_changing": ev.error is None, "result_digest": None, "evidence_refs": [f"adk-event/{ev.event_id}"], "timestamp_ms": _iso_to_ms(ev.timestamp)})
        elif ev.type == "final_response":
            verifiers.append({"verifier_id": "adk-completion", "passed": ev.error is None, "score": 1.0 if ev.error is None else 0.0, "claim_ids": [trace.trace_id]})
    llm_events = [e for e in trace.events if e.type == "model_response"]
    model_id = llm_events[0].content.get("model", trace.model_id) if llm_events else trace.model_id
    root = [e for e in trace.events if e.type in ("function_call", "model_response") and not any(e2.event_id == e.event_id for e2 in trace.events)]
    created_ms = _iso_to_ms(trace.events[0].timestamp) if trace.events else 0.0
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"google-adk/{trace.trace_id}",
        "trace_id": trace.trace_id,
        "model_id": model_id,
        "model_provider": "google",
        "input_refs": [],
        "output_refs": [{"uri": f"google-adk/session/{trace.session_id}"}],
        "capability_decisions": [],
        "actions": actions,
        "verifier_results": verifiers,
        "created_at_ms": created_ms,
    }

def load_adk_trace_jsonl(path: str) -> list:
    import json
    by_trace: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            obj = json.loads(line)
            tid = obj.get("trace_id", "unknown")
            by_trace.setdefault(tid, []).append(obj)
    traces = []
    for trace_id, evs in by_trace.items():
        events = [ADKEvent(event_id=e.get("event_id",""), type=e.get("type",""), timestamp=e.get("timestamp",""), agent_name=e.get("agent_name",""), content=e.get("content",{}), error=e.get("error")) for e in evs]
        llm = [e for e in events if e.type == "model_response"]
        model_id = llm[0].content.get("model","unknown") if llm else "unknown"
        traces.append(ADKTrace(trace_id=trace_id, model_id=model_id, events=events, session_id=evs[0].get("session_id",""), root_agent=evs[0].get("agent_name","")))
    return traces

class GoogleADKAdapter:
    def load_jsonl(self, path: str) -> list: return load_adk_trace_jsonl(path)
    def to_aep(self, traces: list) -> list: return [adk_trace_to_aep(t) for t in traces]
