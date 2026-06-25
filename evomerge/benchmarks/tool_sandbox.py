"""ToolSandbox benchmark adapter. Reference: https://arxiv.org/abs/2408.04682"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TSToolCall:
    tool_name: str
    arguments: dict = field(default_factory=dict)
    result: str = ""
    error: str | None = None
    latency_ms: float = 0.0

@dataclass
class TSTask:
    task_id: str
    user_request: str
    available_tools: list = field(default_factory=list)
    expected_calls: list = field(default_factory=list)
    category: str = "general"

@dataclass
class TSResult:
    task_id: str
    model_id: str
    tool_calls: list  # list[TSToolCall]
    final_response: str = ""
    success: bool = False
    session_id: str = ""

def ts_to_aep(result: TSResult, task: TSTask) -> dict:
    actions = [{"action_id": f"call-{i}", "tool_name": c.tool_name, "state_changing": c.error is None, "result_digest": None, "evidence_refs": [], "timestamp_ms": c.latency_ms} for i, c in enumerate(result.tool_calls)]
    deny_decisions = [{"capability": f"tool:{c.tool_name}", "subject": "agent", "resource": c.tool_name, "decision": "deny", "reason_code": "tool_error"} for c in result.tool_calls if c.error]
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"tool-sandbox/{result.task_id}/{result.model_id}",
        "model_id": result.model_id, "model_provider": "benchmark",
        "input_refs": [{"uri": f"tool-sandbox/task/{result.task_id}"}],
        "output_refs": [{"uri": f"tool-sandbox/result/{result.session_id}"}],
        "capability_decisions": deny_decisions,
        "actions": actions,
        "verifier_results": [{"verifier_id": "tool-sandbox-success", "passed": result.success, "score": 1.0 if result.success else 0.0, "claim_ids": [result.task_id]}],
        "created_at_ms": 0,
    }

def ts_to_rollout(result: TSResult, task: TSTask) -> dict:
    tool_calls = []
    for c in result.tool_calls:
        tool_calls.append({"event": "tool_call", "data": {"name": c.tool_name, "arguments": c.arguments}})
        tool_calls.append({"event": "tool_result", "data": {"result": c.result, "error": c.error}})
    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"tool-sandbox/{result.task_id}",
        "task": task.user_request, "branch_index": 0, "temperature": 0.0,
        "session_id": result.session_id or f"ts-{result.task_id}",
        "tool_call_sequence": tool_calls, "final_answer": result.final_response,
        "build_result": None,
        "objective_score": int(result.success),
        "objective_status": "pass" if result.success else "fail",
        "rank": 0, "total_score": 1.0 if result.success else 0.0,
        "provenance": {"source": "tool-sandbox", "session_id": result.session_id, "job_id": result.task_id, "exported_at_ms": 0, "schema_version": "rollout-wire/v1", "evidence_source": "benchmark_graded", "redaction_version": "none"},
    }

class ToolSandboxAdapter:
    def load_jsonl(self, path: str) -> list:
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                obj = json.loads(line)
                task = TSTask(task_id=obj["task_id"], user_request=obj.get("user_request",""), available_tools=obj.get("available_tools",[]), expected_calls=obj.get("expected_calls",[]), category=obj.get("category","general"))
                result = TSResult(task_id=obj["task_id"], model_id=obj.get("model_id","unknown"), tool_calls=[TSToolCall(**c) for c in obj.get("tool_calls",[])], final_response=obj.get("final_response",""), success=obj.get("success",False), session_id=obj.get("session_id",""))
                pairs.append((result, task))
        return pairs
    def to_aep(self, pairs): return [ts_to_aep(r,t) for r,t in pairs]
    def to_rollouts(self, pairs): return [ts_to_rollout(r,t) for r,t in pairs]
