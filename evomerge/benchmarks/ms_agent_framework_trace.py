"""Microsoft Agent Framework 1.0 trace import.
Reference: https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class MSStep:
    id: str
    name: str
    type: str  # llm|tool|human|condition
    status: str  # completed|failed|skipped
    started_at: str = ""
    ended_at: str = ""
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: str | None = None

@dataclass
class MSWorkflowRun:
    run_id: str
    workflow_name: str
    model_id: str
    steps: list  # list[MSStep]
    status: str = "completed"  # completed|failed|partial
    session_id: str = ""
    agent_name: str = ""

def _iso_to_ms(iso):
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except Exception:
        return 0.0

def ms_workflow_to_aep(run: MSWorkflowRun) -> dict:
    actions = []
    verifiers = []
    cap_decisions = []
    for step in run.steps:
        if step.type == "tool":
            actions.append({"action_id": step.id, "tool_name": step.name, "state_changing": step.status == "completed" and step.error is None, "result_digest": None, "evidence_refs": [f"ms-step/{step.id}"], "timestamp_ms": _iso_to_ms(step.started_at)})
            if step.error or step.status == "failed":
                cap_decisions.append({"capability": f"tool:{step.name}", "subject": run.agent_name or "agent", "resource": step.name, "decision": "deny", "reason_code": "step_failed"})
        elif step.type == "human":
            verifiers.append({"verifier_id": f"human-approval/{step.id}", "passed": step.status == "completed", "score": 1.0 if step.status == "completed" else 0.0, "claim_ids": [step.id]})
    if run.status in ("completed", "failed"):
        verifiers.append({"verifier_id": "ms-workflow-completion", "passed": run.status == "completed", "score": 1.0 if run.status == "completed" else 0.0, "claim_ids": [run.run_id]})
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"ms-agent/{run.run_id}",
        "trace_id": run.run_id,
        "model_id": run.model_id,
        "model_provider": "microsoft",
        "input_refs": [],
        "output_refs": [{"uri": f"ms-agent/run/{run.session_id}"}],
        "capability_decisions": cap_decisions,
        "actions": actions,
        "verifier_results": verifiers,
        "created_at_ms": _iso_to_ms(run.steps[0].started_at) if run.steps else 0.0,
    }

def load_ms_workflow_jsonl(path: str) -> list:
    import json
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            obj = json.loads(line)
            steps = [MSStep(**s) for s in obj.get("steps", [])]
            runs.append(MSWorkflowRun(run_id=obj["run_id"], workflow_name=obj.get("workflow_name",""), model_id=obj.get("model_id","unknown"), steps=steps, status=obj.get("status","completed"), session_id=obj.get("session_id",""), agent_name=obj.get("agent_name","")))
    return runs

class MSAgentFrameworkAdapter:
    def load_jsonl(self, path: str) -> list: return load_ms_workflow_jsonl(path)
    def to_aep(self, runs: list) -> list: return [ms_workflow_to_aep(r) for r in runs]
