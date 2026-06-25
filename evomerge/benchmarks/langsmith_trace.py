"""Cross-framework trace import: LangSmith/LangGraph trace to AEP record.

LangSmith exports runs as JSONL where each line is one Run object.
Runs are linked via parent_run_id forming a tree.
run_type: "chain" | "llm" | "tool" | "retriever" | "embedding" | "prompt"

This module converts a LangSmith run tree to an AEP record so
WasmAgent evidence scoring and benchmark audit can operate on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LSRun:
    """One LangSmith run."""
    id: str
    name: str
    run_type: str          # chain | llm | tool | retriever | embedding | prompt
    start_time: str        # ISO-8601
    end_time: str          # ISO-8601
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    parent_run_id: str | None = None
    extra: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)


@dataclass
class LSTrace:
    """A full LangSmith execution trace (tree of runs)."""
    trace_id: str         # root run id
    runs: list            # list[LSRun], all runs in this trace
    model_id: str = ""
    session_id: str = ""


def _iso_to_ms(iso: str) -> float:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except (ValueError, OSError):
        return 0.0


def ls_trace_to_aep(trace: LSTrace) -> dict:
    """Convert a LangSmith LSTrace to an AEP record dict."""
    actions = []
    capability_decisions = []
    model_id = trace.model_id

    # Find root chain run for created_at_ms and run_id
    root_runs = [r for r in trace.runs if r.run_type == "chain" and r.parent_run_id is None]
    created_at_ms = _iso_to_ms(root_runs[0].start_time) if root_runs else 0.0

    for run in trace.runs:
        if run.run_type == "tool":
            state_changing = run.error is None and bool(run.outputs)
            actions.append({
                "action_id": run.id,
                "tool_name": run.name,
                "state_changing": state_changing,
                "precondition_digest": None,
                "result_digest": None,
                "evidence_refs": [f"langsmith/{run.id}"],
                "timestamp_ms": _iso_to_ms(run.start_time),
            })
            if run.error:
                capability_decisions.append({
                    "capability": f"tool:{run.name}",
                    "subject": "agent",
                    "resource": run.name,
                    "decision": "deny",
                    "reason_code": "tool_error",
                })
        elif run.run_type == "chain" and run.error:
            capability_decisions.append({
                "capability": f"chain:{run.name}",
                "subject": "agent",
                "resource": run.name,
                "decision": "deny",
                "reason_code": "chain_error",
            })
        elif run.run_type == "llm":
            # Extract model_id from extra.invocation_params.model, or fall back to run name
            invocation_params = run.extra.get("invocation_params", {})
            candidate = invocation_params.get("model", "") or run.name
            if candidate:
                model_id = candidate

    return {
        "schema_version": "aep/v0.1",
        "run_id": f"langsmith/{trace.trace_id}",
        "trace_id": trace.trace_id,
        "model_id": model_id,
        "model_provider": "langsmith",
        "input_refs": [],
        "output_refs": [{"uri": f"langsmith/session/{trace.session_id}"}],
        "capability_decisions": capability_decisions,
        "actions": actions,
        "verifier_results": [],
        "created_at_ms": created_at_ms,
    }


def load_ls_trace_jsonl(path: str) -> list:
    """Load a LangSmith JSONL export and return a list of LSTrace objects.

    Each line is one Run. Runs with no parent_run_id are trace roots; runs
    with a parent_run_id are attached to their parent trace by following the
    parent_run_id chain back to the root.
    """
    raw_runs: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_runs.append(json.loads(line))

    # Build id → raw dict for quick parent lookup
    by_id: dict[str, dict] = {r["id"]: r for r in raw_runs}

    def _root_id(run: dict) -> str:
        """Walk parent_run_id chain to find the root run id."""
        current = run
        while current.get("parent_run_id") is not None:
            parent_id = current["parent_run_id"]
            if parent_id not in by_id:
                # Parent not present in file — treat current as root
                break
            current = by_id[parent_id]
        return current["id"]

    # Group all runs by their root run id
    by_root: dict[str, list[dict]] = {}
    for r in raw_runs:
        rid = _root_id(r)
        by_root.setdefault(rid, []).append(r)

    traces = []
    for root_run_id, run_dicts in by_root.items():
        ls_runs = [
            LSRun(
                id=r.get("id", ""),
                name=r.get("name", ""),
                run_type=r.get("run_type", "chain"),
                start_time=r.get("start_time", ""),
                end_time=r.get("end_time", ""),
                inputs=r.get("inputs", {}),
                outputs=r.get("outputs", {}),
                error=r.get("error"),
                parent_run_id=r.get("parent_run_id"),
                extra=r.get("extra", {}),
                tags=r.get("tags", []),
            )
            for r in run_dicts
        ]
        # Derive model_id from llm runs in this trace
        model_id = ""
        for run in ls_runs:
            if run.run_type == "llm":
                params = run.extra.get("invocation_params", {})
                model_id = params.get("model", "") or run.name or model_id

        root_dict = by_id.get(root_run_id, {})
        traces.append(LSTrace(
            trace_id=root_run_id,
            runs=ls_runs,
            model_id=model_id,
            session_id=root_dict.get("session_id", ""),
        ))
    return traces
