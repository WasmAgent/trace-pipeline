"""AgentHarm / OS-Harm / CUAHarm safety benchmark adapter.
References: https://arxiv.org/abs/2410.09024
Evaluates whether agents perform harmful actions when instructed.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class HarmAction:
    action_id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)
    is_harmful: bool = False
    was_prevented: bool = False
    harm_category: str = ""  # violence|fraud|privacy|malware|other

@dataclass
class HarmTask:
    task_id: str
    harmful_instruction: str
    harm_category: str
    severity: str = "medium"  # low|medium|high|critical
    benchmark: str = "agent-harm"  # agent-harm|os-harm|cua-harm

@dataclass
class HarmResult:
    task_id: str
    model_id: str
    actions: list  # list[HarmAction]
    refused: bool = False  # did the agent refuse the instruction?
    harmful_actions_taken: int = 0
    session_id: str = ""

def harm_to_aep(result: HarmResult, task: HarmTask) -> dict:
    aep_actions = [{"action_id": a.action_id, "tool_name": a.tool_name, "state_changing": not a.was_prevented, "result_digest": None, "evidence_refs": [], "timestamp_ms": 0.0} for a in result.actions]
    cap_decisions = []
    for a in result.actions:
        if a.was_prevented:
            cap_decisions.append({"capability": f"tool:{a.tool_name}", "subject": "agent", "resource": a.tool_name, "decision": "deny", "reason_code": f"harm_prevention:{a.harm_category}"})
    safe = result.refused or result.harmful_actions_taken == 0
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"{task.benchmark}/{result.task_id}/{result.model_id}",
        "model_id": result.model_id, "model_provider": "benchmark",
        "input_refs": [{"uri": f"{task.benchmark}/task/{result.task_id}"}],
        "output_refs": [{"uri": f"{task.benchmark}/result/{result.session_id}"}],
        "capability_decisions": cap_decisions,
        "actions": aep_actions,
        "verifier_results": [{"verifier_id": f"{task.benchmark}-safety", "passed": safe, "score": 1.0 if safe else 0.0, "claim_ids": [f"harm-prevention:{task.harm_category}"]}],
        "created_at_ms": 0,
    }

def harm_to_rollout(result: HarmResult, task: HarmTask) -> dict:
    tool_calls = []
    for a in result.actions:
        tool_calls.append({"event": "tool_call", "data": {"name": a.tool_name, "arguments": a.arguments}})
        tool_calls.append({"event": "tool_result", "data": {"result": "prevented" if a.was_prevented else "executed"}})
    safe = result.refused or result.harmful_actions_taken == 0
    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"{task.benchmark}/{result.task_id}",
        "task": task.harmful_instruction, "branch_index": 0, "temperature": 0.0,
        "session_id": result.session_id or f"harm-{result.task_id}",
        "tool_call_sequence": tool_calls, "final_answer": "refused" if result.refused else "",
        "build_result": None,
        "objective_score": int(safe),
        "objective_status": "pass" if safe else "fail",
        "rank": 0, "total_score": 1.0 if safe else 0.0,
        "provenance": {"source": task.benchmark, "session_id": result.session_id, "job_id": result.task_id, "exported_at_ms": 0, "schema_version": "rollout-wire/v1", "evidence_source": "benchmark_graded", "redaction_version": "none"},
    }

class AgentHarmAdapter:
    def load_jsonl(self, path: str) -> list:
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                obj = json.loads(line)
                task = HarmTask(task_id=obj["task_id"], harmful_instruction=obj.get("harmful_instruction",""), harm_category=obj.get("harm_category","other"), severity=obj.get("severity","medium"), benchmark=obj.get("benchmark","agent-harm"))
                result = HarmResult(task_id=obj["task_id"], model_id=obj.get("model_id","unknown"), actions=[HarmAction(**a) for a in obj.get("actions",[])], refused=obj.get("refused",False), harmful_actions_taken=obj.get("harmful_actions_taken",0), session_id=obj.get("session_id",""))
                pairs.append((result, task))
        return pairs
    def to_aep(self, pairs): return [harm_to_aep(r,t) for r,t in pairs]
    def to_rollouts(self, pairs): return [harm_to_rollout(r,t) for r,t in pairs]
