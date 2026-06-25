"""Generate wasmagent-smoke-traces-v1 synthetic dataset.

Creates 5 AEP records and 3 rollout-wire/v1 records for pipeline testing.
All data is fully synthetic — no real user tasks, no real API keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


AEP_TASKS = [
    ("smoke-run-001", "task://implement-fibonacci", "claude-sonnet-4-6", "anthropic", True),
    ("smoke-run-002", "task://reverse-string", "claude-haiku-4-5-20251001", "anthropic", True),
    ("smoke-run-003", "task://sort-array", "claude-sonnet-4-6", "anthropic", True),
    ("smoke-run-004", "task://prompt-injection-blocked", "claude-sonnet-4-6", "anthropic", False),
    ("smoke-run-005", "task://write-unit-test", "claude-haiku-4-5-20251001", "anthropic", True),
]

aep_records = []
for i, (run_id, task_uri, model_id, provider, passed) in enumerate(AEP_TASKS):
    base_ms = 1750900000000 + i * 3000
    record = {
        "schema_version": "aep/v0.1",
        "run_id": run_id,
        "trace_id": sha256(run_id)[:32],
        "parent_trace_id": None,
        "repo_commit": "smoke-fixture-v0.1.0",
        "runtime_version": "1.0.3",
        "model_provider": provider,
        "model_id": model_id,
        "policy_bundle_digest": sha256("wasmagent-default-1.0.0"),
        "tool_manifest_digest": sha256(f"tools-{run_id}"),
        "input_refs": [{"uri": task_uri, "taint_labels": []}],
        "output_refs": [
            {"uri": "answer://final", "digest": sha256(f"answer-{run_id}")}
        ],
        "capability_decisions": [
            {
                "capability": "execute_code",
                "subject": f"agent:{run_id}",
                "resource": "kernel:quickjs",
                "decision": "allow" if passed else "deny",
                "reason_code": "policy:default-allow-execute" if passed else "policy:deny-blocked-vetting",
            }
        ],
        "actions": [
            {
                "action_id": f"act-{run_id}-0",
                "tool_name": "execute_code",
                "state_changing": False,
                "result_digest": sha256(f"result-{run_id}-0"),
                "evidence_refs": [],
                "timestamp_ms": base_ms,
            }
        ] if passed else [],
        "verifier_results": [
            {
                "verifier_id": "build-passes",
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "claim_ids": ["output-is-correct"],
            }
        ],
        "budget_ledger": {
            "token_budget": {"limit": 4096, "spent": 200 + i * 50},
            "latency_budget": {"limit_ms": 30000, "actual_ms": 800 + i * 200},
            "tool_budget": {"limit": 10, "spent": 1 if passed else 0},
        },
        "created_at_ms": base_ms + 1000,
    }
    aep_records.append(record)

aep_path = OUT / "aep-smoke.jsonl"
with open(aep_path, "w") as f:
    for r in aep_records:
        f.write(json.dumps(r) + "\n")
print(f"Written {len(aep_records)} AEP records → {aep_path}")


ROLLOUT_TASKS = [
    ("rollout-smoke-001", "implement a fibonacci function in JavaScript", 1),
    ("rollout-smoke-002", "write a function that reverses a string", 1),
    ("rollout-smoke-003", "sort an array of numbers in ascending order", 0),
]

rollout_records = []
for i, (rollout_id, task, obj_score) in enumerate(ROLLOUT_TASKS):
    base_ms = 1750910000000 + i * 5000
    record = {
        "schema_version": "rollout-wire/v1",
        "rollout_id": rollout_id,
        "task": task,
        "branch_index": 0,
        "temperature": 0.7,
        "session_id": f"session-smoke-{i:03d}",
        "tool_call_sequence": [
            {
                "tool_name": "execute_code",
                "args": {"language": "javascript", "code": f"// Solution for: {task}"},
                "result": {"exit_code": 0 if obj_score else 1, "stdout": ""},
                "timestamp_ms": base_ms,
                "state_changing": False,
            }
        ],
        "final_answer": f"Solution for '{task}' — smoke fixture",
        "build_result": {
            "exit_code": 0 if obj_score else 1,
            "stdout": "Build passed" if obj_score else "Build failed",
            "stderr": "",
            "timestamp_ms": base_ms + 500,
        },
        "objective_score": obj_score,
        "objective_status": "pass" if obj_score else "fail",
        "rank": 1 if obj_score else 2,
        "total_score": float(obj_score),
        "provenance": {
            "source": "bscode-smoke",
            "session_id": f"session-smoke-{i:03d}",
            "created_at_ms": base_ms + 1000,
            "model_id": "claude-sonnet-4-6",
            "model_provider": "anthropic",
            "repo_commit": "smoke-fixture-v0.1.0",
            "runtime_version": "1.0.3",
            "policy_bundle_digest": sha256("wasmagent-default-1.0.0"),
        },
    }
    rollout_records.append(record)

rollout_path = OUT / "rollout-smoke.jsonl"
with open(rollout_path, "w") as f:
    for r in rollout_records:
        f.write(json.dumps(r) + "\n")
print(f"Written {len(rollout_records)} rollout records → {rollout_path}")
print("Done. Run: python3 -m evomerge validate-aep --input data/smoke/aep-smoke.jsonl")
