"""Terminal-Bench benchmark adapter.
Reference: https://arxiv.org/abs/2601.11868
Evaluates agents on real terminal tasks with human-written solutions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TBCommand:
    command: str
    output: str = ""
    exit_code: int = 0
    timestamp_ms: float = 0.0

@dataclass
class TBTask:
    task_id: str
    description: str
    category: str = "general"  # file_edit | shell | git | build | general
    setup_script: str = ""
    solution_tests: list = field(default_factory=list)

@dataclass
class TBResult:
    task_id: str
    model_id: str
    trajectory: list  # list[TBCommand]
    passed: bool = False
    latency_ms: float = 0.0
    session_id: str = ""

def tb_to_aep(result: TBResult, task: TBTask) -> dict:
    actions = []
    for i, cmd in enumerate(result.trajectory):
        actions.append({
            "action_id": f"cmd-{i}",
            "tool_name": "bash",
            "state_changing": cmd.exit_code == 0 and bool(cmd.output),
            "result_digest": None,
            "evidence_refs": [f"tb-cmd/{i}"],
            "timestamp_ms": cmd.timestamp_ms,
        })
    return {
        "schema_version": "aep/v0.1",
        "run_id": f"terminal-bench/{result.task_id}/{result.model_id}",
        "model_id": result.model_id,
        "model_provider": "benchmark",
        "input_refs": [{"uri": f"terminal-bench/task/{result.task_id}"}],
        "output_refs": [{"uri": f"terminal-bench/result/{result.session_id}"}],
        "capability_decisions": [],
        "actions": actions,
        "verifier_results": [{
            "verifier_id": "terminal-bench-tests",
            "passed": result.passed,
            "score": 1.0 if result.passed else 0.0,
            "claim_ids": [result.task_id],
        }],
        "created_at_ms": 0,
    }

def tb_to_rollout(result: TBResult, task: TBTask) -> dict:
    tool_calls = []
    for cmd in result.trajectory:
        tool_calls.append({"event": "tool_call", "data": {"name": "bash", "arguments": {"command": cmd.command}}})
        tool_calls.append({"event": "tool_result", "data": {"result": cmd.output, "exit_code": cmd.exit_code}})
    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"terminal-bench/{result.task_id}",
        "task": task.description,
        "branch_index": 0,
        "temperature": 0.0,
        "session_id": result.session_id or f"tb-{result.task_id}",
        "tool_call_sequence": tool_calls,
        "final_answer": "",
        "build_result": {"status": "pass" if result.passed else "fail", "exit_code": 0 if result.passed else 1, "stderr": ""},
        "objective_score": int(result.passed),
        "objective_status": "pass" if result.passed else "fail",
        "rank": 0,
        "total_score": 1.0 if result.passed else 0.0,
        "provenance": {"source": "terminal-bench", "session_id": result.session_id, "job_id": result.task_id, "exported_at_ms": 0, "schema_version": "rollout-wire/v1", "evidence_source": "benchmark_graded", "redaction_version": "none"},
    }

class TerminalBenchAdapter:
    def load_jsonl(self, path: str) -> list:
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                obj = json.loads(line)
                task = TBTask(task_id=obj["task_id"], description=obj.get("description",""), category=obj.get("category","general"), setup_script=obj.get("setup_script",""), solution_tests=obj.get("solution_tests",[]))
                result = TBResult(task_id=obj["task_id"], model_id=obj.get("model_id","unknown"), trajectory=[TBCommand(**c) for c in obj.get("trajectory",[])], passed=obj.get("passed",False), latency_ms=obj.get("latency_ms",0.0), session_id=obj.get("session_id",""))
                pairs.append((result, task))
        return pairs
    def to_aep(self, pairs): return [tb_to_aep(r,t) for r,t in pairs]
    def to_rollouts(self, pairs): return [tb_to_rollout(r,t) for r,t in pairs]
