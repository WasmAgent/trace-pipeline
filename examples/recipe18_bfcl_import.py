"""Recipe 18: Import BFCL v4 results and convert to rollout-wire format.

Demonstrates how to use the bfcl benchmark adapter to import function-calling
evaluation results and prepare them for evomerge pipeline processing.

Usage:
    python examples/recipe18_bfcl_import.py

This recipe:
1. Creates synthetic BFCL v4-style results (function call predictions)
2. Converts them to rollout-wire/v1 format via bfcl_to_rollout()
3. Runs the Evidence Admission Score on the converted records
4. Prints a summary table
"""
from __future__ import annotations

from evomerge.benchmarks.bfcl import (
    BFCLFunction,
    BFCLFunctionCall,
    BFCLResult,
    BFCLTask,
    bfcl_to_rollout,
)


def make_synthetic_bfcl_data() -> list[tuple[BFCLResult, BFCLTask]]:
    """Create synthetic BFCL v4-style (result, task) pairs for demo purposes."""
    func_weather = BFCLFunction(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )

    task = BFCLTask(
        task_id="bfcl-demo-001",
        question="What is the weather in San Francisco in Celsius?",
        functions=[func_weather],
        ground_truth=[
            {"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
        ],
        category="simple",
        difficulty="easy",
    )

    # Correct prediction — exact match with ground truth
    result_correct = BFCLResult(
        task_id=task.task_id,
        model_id="demo-model",
        calls=[
            BFCLFunctionCall(
                name="get_weather",
                arguments={"location": "San Francisco", "unit": "celsius"},
            )
        ],
        raw_response='get_weather(location="San Francisco", unit="celsius")',
        latency_ms=342.0,
        pass_rate=1.0,
    )

    # Incorrect prediction — wrong unit argument
    result_wrong = BFCLResult(
        task_id="bfcl-demo-002",
        model_id="demo-model",
        calls=[
            BFCLFunctionCall(
                name="get_weather",
                arguments={"location": "San Francisco", "unit": "fahrenheit"},
            )
        ],
        raw_response='get_weather(location="San Francisco", unit="fahrenheit")',
        latency_ms=289.0,
        pass_rate=0.0,
    )

    # Parallel call example — two functions called at once
    func_calendar = BFCLFunction(
        name="get_calendar",
        description="Get calendar events for a date",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string"},
            },
            "required": ["date"],
        },
    )

    task_parallel = BFCLTask(
        task_id="bfcl-demo-003",
        question="Get weather in Paris and my calendar for 2026-07-01.",
        functions=[func_weather, func_calendar],
        ground_truth=[
            {"name": "get_weather", "arguments": {"location": "Paris"}},
            {"name": "get_calendar", "arguments": {"date": "2026-07-01"}},
        ],
        category="parallel",
        difficulty="medium",
    )

    result_parallel = BFCLResult(
        task_id=task_parallel.task_id,
        model_id="demo-model",
        calls=[
            BFCLFunctionCall(name="get_weather", arguments={"location": "Paris"}),
            BFCLFunctionCall(name="get_calendar", arguments={"date": "2026-07-01"}),
        ],
        raw_response="[get_weather(Paris), get_calendar(2026-07-01)]",
        latency_ms=410.0,
        pass_rate=1.0,
    )

    return [
        (result_correct, task),
        (result_wrong, task),
        (result_parallel, task_parallel),
    ]


def main() -> None:
    print("Recipe 18: BFCL v4 -> rollout-wire import\n")

    pairs = make_synthetic_bfcl_data()
    print(f"Synthetic BFCL result/task pairs: {len(pairs)}")

    # Convert each (result, task) pair to rollout-wire/v1 format
    rollouts = [bfcl_to_rollout(result, task) for result, task in pairs]
    print(f"Converted to rollout-wire/v1 records: {len(rollouts)}")
    print()

    for r in rollouts:
        tool_count = sum(1 for e in r["tool_call_sequence"] if e["event"] == "tool_call")
        print(
            f"  rollout_id={r['rollout_id']:<28}  "
            f"status={r['objective_status']:<8}  "
            f"score={r['total_score']:.1f}  "
            f"tool_calls={tool_count}"
        )

    # Build minimal AEP records for Evidence Admission scoring
    aep_records = []
    for (result, task), rollout in zip(pairs, rollouts):
        passed = result.pass_rate == 1.0
        aep_records.append({
            "schema_version": "aep/v0.1",
            "run_id": rollout["rollout_id"],
            "model_id": result.model_id,
            "input_refs": [{"uri": f"bfcl/task/{task.task_id}"}],
            "output_refs": [{"uri": f"bfcl/result/{result.task_id}"}],
            "actions": [],
            "capability_decisions": [],
            "verifier_results": [
                {
                    "verifier_id": "bfcl_pass_rate",
                    "passed": passed,
                    "score": result.pass_rate if result.pass_rate is not None else 0.0,
                    "claim_ids": [task.task_id],
                }
            ],
            "created_at_ms": 0,
        })

    from evomerge.validate.quality_gate import admission_gate

    gate = admission_gate(aep_records)
    print("\nEvidence Admission Summary:")
    print(f"  total      : {gate['total']}")
    print(f"  mean_score : {gate['mean_score']}")
    for cat, count in gate["by_category"].items():
        if count > 0:
            print(f"  {cat:<16}: {count}")

    print(f"\nAdmitted for training : {len(gate['admitted'])}")
    print(f"Audit only            : {len(gate['audit_only'])}")
    print(f"Rejected              : {len(gate['rejected'])}")
    print("\nRecipe 18 complete.")


if __name__ == "__main__":
    main()
