"""A2A (Agent-to-Agent) task trace import.
Reference: https://a2a-protocol.org/latest/
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class A2AMessage:
    message_id: str
    role: str  # "user"|"agent"
    content: str
    agent_name: str = ""
    timestamp: str = ""

@dataclass
class A2AArtifact:
    artifact_id: str
    name: str
    content_type: str = ""
    uri: str = ""

@dataclass
class A2ATask:
    task_id: str
    session_id: str
    messages: list  # list[A2AMessage]
    artifacts: list = field(default_factory=list)  # list[A2AArtifact]
    status: str = "completed"  # submitted|working|input-required|completed|failed|canceled
    model_id: str = "unknown"

def _iso_to_ms(iso):
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except Exception:
        return 0.0

def a2a_task_to_aep(task: A2ATask) -> dict:
    agent_messages = [m for m in task.messages if m.role == "agent"]
    actions = [{"action_id": m.message_id, "tool_name": "a2a_message", "state_changing": False, "result_digest": None, "evidence_refs": [f"a2a-msg/{m.message_id}"], "timestamp_ms": _iso_to_ms(m.timestamp)} for m in agent_messages]
    output_refs = [{"uri": a.uri or f"a2a-artifact/{a.artifact_id}", "redaction_profile": None} for a in task.artifacts]
    passed = task.status == "completed"
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"a2a/{task.task_id}",
        "trace_id": task.session_id,
        "model_id": task.model_id,
        "model_provider": "a2a",
        "input_refs": [],
        "output_refs": output_refs or [{"uri": f"a2a/session/{task.session_id}"}],
        "capability_decisions": [],
        "actions": actions,
        "verifier_results": [{"verifier_id": "a2a-task-completion", "passed": passed, "score": 1.0 if passed else 0.0, "claim_ids": [task.task_id]}],
        "created_at_ms": _iso_to_ms(task.messages[0].timestamp) if task.messages else 0.0,
    }

def load_a2a_task_jsonl(path: str) -> list:
    import json
    tasks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            obj = json.loads(line)
            messages = [A2AMessage(**m) for m in obj.get("messages", [])]
            artifacts = [A2AArtifact(**a) for a in obj.get("artifacts", [])]
            tasks.append(A2ATask(task_id=obj["task_id"], session_id=obj.get("session_id",""), messages=messages, artifacts=artifacts, status=obj.get("status","completed"), model_id=obj.get("model_id","unknown")))
    return tasks

class A2ATaskAdapter:
    def load_jsonl(self, path: str) -> list: return load_a2a_task_jsonl(path)
    def to_aep(self, tasks: list) -> list: return [a2a_task_to_aep(t) for t in tasks]
