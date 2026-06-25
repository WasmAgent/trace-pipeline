"""τ-bench / τ³-bench adapter. Reference: https://github.com/sierra-research/tau-bench"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TauTurn:
    role: str  # "user" | "agent" | "tool"
    content: str
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None

@dataclass
class TauTask:
    task_id: str
    user_query: str
    rules: list = field(default_factory=list)
    expected_actions: list = field(default_factory=list)
    category: str = "retail"  # retail | airline | etc.

@dataclass
class TauResult:
    task_id: str
    model_id: str
    turns: list  # list[TauTurn]
    rule_violations: int = 0
    task_completed: bool = False
    session_id: str = ""

def tau_to_aep(result: TauResult, task: TauTask) -> dict:
    actions = [{"action_id": f"turn-{i}", "tool_name": t.tool_name or "chat", "state_changing": t.tool_name is not None, "evidence_refs": [], "timestamp_ms": float(i * 1000)} for i, t in enumerate(result.turns) if t.role == "agent"]
    verifiers = [{"verifier_id": "tau-rule-compliance", "passed": result.rule_violations == 0, "score": max(0.0, 1.0 - result.rule_violations * 0.2), "claim_ids": [f"rule-{j}" for j in range(len(task.rules))]}]
    if result.task_completed or not result.task_completed:
        verifiers.append({"verifier_id": "tau-task-completion", "passed": result.task_completed, "score": 1.0 if result.task_completed else 0.0, "claim_ids": [result.task_id]})
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"tau-bench/{result.task_id}/{result.model_id}",
        "model_id": result.model_id, "model_provider": "benchmark",
        "input_refs": [{"uri": f"tau-bench/task/{result.task_id}"}],
        "output_refs": [{"uri": f"tau-bench/result/{result.session_id}"}],
        "capability_decisions": [],
        "actions": actions,
        "verifier_results": verifiers,
        "created_at_ms": 0,
    }

def tau_to_rollout(result: TauResult, task: TauTask) -> dict:
    tool_calls = []
    for t in result.turns:
        if t.role == "agent" and t.tool_name:
            tool_calls.append({"event": "tool_call", "data": {"name": t.tool_name, "arguments": t.tool_args}})
        elif t.role == "tool":
            tool_calls.append({"event": "tool_result", "data": {"result": t.tool_result or ""}})
    passed = result.task_completed and result.rule_violations == 0
    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"tau-bench/{result.task_id}",
        "task": task.user_query, "branch_index": 0, "temperature": 0.0,
        "session_id": result.session_id or f"tau-{result.task_id}",
        "tool_call_sequence": tool_calls, "final_answer": "",
        "build_result": None,
        "objective_score": int(passed),
        "objective_status": "pass" if passed else "fail",
        "rank": 0, "total_score": 1.0 if passed else 0.0,
        "provenance": {"source": "tau-bench", "session_id": result.session_id, "job_id": result.task_id, "exported_at_ms": 0, "schema_version": "rollout-wire/v1", "evidence_source": "benchmark_graded", "redaction_version": "none"},
    }

class TauBenchAdapter:
    def load_jsonl(self, path: str) -> list:
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                obj = json.loads(line)
                task = TauTask(task_id=obj["task_id"], user_query=obj.get("user_query",""), rules=obj.get("rules",[]), expected_actions=obj.get("expected_actions",[]), category=obj.get("category","retail"))
                result = TauResult(task_id=obj["task_id"], model_id=obj.get("model_id","unknown"), turns=[TauTurn(**t) for t in obj.get("turns",[])], rule_violations=obj.get("rule_violations",0), task_completed=obj.get("task_completed",False), session_id=obj.get("session_id",""))
                pairs.append((result, task))
        return pairs
    def to_aep(self, pairs): return [tau_to_aep(r,t) for r,t in pairs]
    def to_rollouts(self, pairs): return [tau_to_rollout(r,t) for r,t in pairs]
