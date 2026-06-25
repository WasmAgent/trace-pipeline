"""BFCL v4 (Berkeley Function Calling Leaderboard) benchmark adapter.

Schema reference: https://gorilla.cs.berkeley.edu/leaderboard.html
BFCL v4 extends v3 with agentic evaluation (multi-turn, state-tracking).

This module defines:
- BFCLFunction: schema for one function declaration in a BFCL task
- BFCLTask: schema for one BFCL task (function calling evaluation)
- BFCLResult: schema for one model response
- BFCLAdapterRecord: converted to internal rollout-wire format
- bfcl_to_rollout(): converts BFCLResult → RolloutBranchRecord-compatible dict
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BFCLFunction:
    """One function declaration in a BFCL task."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class BFCLTask:
    """One BFCL v4 evaluation task."""
    task_id: str
    question: str                          # user query
    functions: list[BFCLFunction]          # available functions
    ground_truth: list[dict[str, Any]]     # expected function calls
    category: str = "simple"              # simple | parallel | multiple | nested | agentic
    difficulty: str = "easy"              # easy | medium | hard


@dataclass
class BFCLFunctionCall:
    """One function call made by the model."""
    name: str
    arguments: dict[str, Any]


@dataclass
class BFCLResult:
    """One model response to a BFCL task."""
    task_id: str
    model_id: str
    calls: list[BFCLFunctionCall]
    raw_response: str = ""
    latency_ms: float = 0.0
    pass_rate: float | None = None         # 1.0 = correct, 0.0 = wrong, None = ungraded


def bfcl_to_rollout(result: BFCLResult, task: BFCLTask) -> dict[str, Any]:
    """Convert a BFCLResult + BFCLTask to a rollout-wire/v1-compatible dict.

    The resulting dict can be validated against rollout-wire.schema.json and
    imported into trace-pipeline via the standard export pipeline.
    """
    tool_calls = []
    for call in result.calls:
        tool_calls.append({
            "event": "tool_call",
            "data": {"name": call.name, "arguments": call.arguments},
        })
        # Synthetic result — grader would fill this
        tool_calls.append({
            "event": "tool_result",
            "data": {"result": f"[bfcl-grader: {call.name}]"},
        })

    objective_score = int(result.pass_rate or 0)
    objective_status = (
        "pass" if result.pass_rate == 1.0
        else "fail" if result.pass_rate == 0.0
        else "unknown"
    )

    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"bfcl/{result.task_id}",
        "task": task.question,
        "branch_index": 0,
        "temperature": 0.0,
        "session_id": f"bfcl-{result.task_id}-{result.model_id}",
        "tool_call_sequence": tool_calls,
        "final_answer": result.raw_response,
        "build_result": None,
        "objective_score": objective_score,
        "objective_status": objective_status,
        "rank": 0,
        "total_score": float(result.pass_rate or 0),
        "provenance": {
            "source": "bfcl",
            "session_id": f"bfcl-{result.task_id}-{result.model_id}",
            "job_id": result.task_id,
            "exported_at_ms": 0,
            "schema_version": "rollout-wire/v1",
            "evidence_source": "benchmark_graded",
            "redaction_version": "none",
        },
    }


class BFCLAdapter:
    """Adapter for BFCL v4 data import.

    Usage:
        adapter = BFCLAdapter()
        records = adapter.load_jsonl("bfcl_results.jsonl")
        rollouts = adapter.to_rollouts(records)
    """

    def load_jsonl(self, path: str) -> list[tuple[BFCLResult, BFCLTask]]:
        """Load BFCL results+tasks from a JSONL file.

        Each line should be a JSON object with keys matching BFCLResult + BFCLTask.
        Returns list of (result, task) tuples.
        """
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                obj = json.loads(line)
                task = BFCLTask(
                    task_id=obj["task_id"],
                    question=obj["question"],
                    functions=[BFCLFunction(**f) for f in obj.get("functions", [])],
                    ground_truth=obj.get("ground_truth", []),
                    category=obj.get("category", "simple"),
                    difficulty=obj.get("difficulty", "easy"),
                )
                result = BFCLResult(
                    task_id=obj["task_id"],
                    model_id=obj.get("model_id", "unknown"),
                    calls=[BFCLFunctionCall(**c) for c in obj.get("calls", [])],
                    raw_response=obj.get("raw_response", ""),
                    latency_ms=obj.get("latency_ms", 0.0),
                    pass_rate=obj.get("pass_rate"),
                )
                pairs.append((result, task))
        return pairs

    def to_rollouts(self, pairs: list[tuple[BFCLResult, BFCLTask]]) -> list[dict]:
        return [bfcl_to_rollout(r, t) for r, t in pairs]
