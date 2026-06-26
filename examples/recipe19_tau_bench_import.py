"""Recipe 19: Import tau-bench results and convert to rollout-wire format.

Demonstrates how to use the tau_bench adapter to import task-and-user
simulation results and prepare them for evomerge pipeline processing.

Usage:
    python examples/recipe19_tau_bench_import.py

This recipe:
1. Creates synthetic tau-bench-style (result, task) pairs
2. Converts them to rollout-wire/v1 format via tau_to_rollout()
3. Also converts to AEP format via tau_to_aep()
4. Runs the Evidence Admission Score
5. Prints a summary table
"""
from __future__ import annotations

from evomerge.benchmarks.tau_bench import (
    TauResult,
    TauTask,
    TauTurn,
    tau_to_aep,
    tau_to_rollout,
)


def make_synthetic_tau_data() -> list[tuple[TauResult, TauTask]]:
    """Create synthetic tau-bench (result, task) pairs for demo purposes."""

    # Task 1: successful retail return
    task_return = TauTask(
        task_id="tau-demo-001",
        user_query="I want to return my order #12345 and get a refund.",
        rules=[
            "Returns must be initiated within 30 days of delivery.",
            "Refunds are processed in 5-7 business days.",
            "Customer must confirm shipping address before refund is issued.",
        ],
        expected_actions=["lookup_order", "initiate_return", "issue_refund"],
        category="retail",
    )

    result_success = TauResult(
        task_id=task_return.task_id,
        model_id="demo-model",
        turns=[
            TauTurn(role="user", content="I want to return order #12345."),
            TauTurn(
                role="agent",
                content="Let me look up your order.",
                tool_name="lookup_order",
                tool_args={"order_id": "12345"},
            ),
            TauTurn(role="tool", content="", tool_result='{"order_id": "12345", "status": "delivered"}'),
            TauTurn(
                role="agent",
                content="Initiating return for order #12345.",
                tool_name="initiate_return",
                tool_args={"order_id": "12345"},
            ),
            TauTurn(role="tool", content="", tool_result='{"return_id": "R-9001", "status": "pending"}'),
            TauTurn(
                role="agent",
                content="Processing refund.",
                tool_name="issue_refund",
                tool_args={"return_id": "R-9001"},
            ),
            TauTurn(role="tool", content="", tool_result='{"refund_id": "REF-001", "amount": 49.99}'),
            TauTurn(role="agent", content="Your refund of $49.99 has been issued."),
        ],
        rule_violations=0,
        task_completed=True,
        session_id="tau-session-001",
    )

    # Task 2: failed airline rebooking (incomplete, rule violation)
    task_flight = TauTask(
        task_id="tau-demo-002",
        user_query="Rebook my flight AA123 to depart on 2026-07-10.",
        rules=[
            "Same-day rebooking fee applies for changes within 24 hours.",
            "Must confirm passenger identity before ticket changes.",
        ],
        expected_actions=["verify_identity", "lookup_flight", "rebook_flight"],
        category="airline",
    )

    result_failed = TauResult(
        task_id=task_flight.task_id,
        model_id="demo-model",
        turns=[
            TauTurn(role="user", content="Please rebook flight AA123 to 2026-07-10."),
            TauTurn(
                role="agent",
                content="Looking up the flight directly.",
                tool_name="lookup_flight",
                tool_args={"flight_id": "AA123"},
            ),
            TauTurn(role="tool", content="", tool_result='{"flight": "AA123", "seats": 12}'),
            TauTurn(role="agent", content="I cannot complete the rebooking without identity verification."),
        ],
        rule_violations=1,       # skipped identity verification
        task_completed=False,
        session_id="tau-session-002",
    )

    # Task 3: successful chat-only query (no tool calls)
    task_info = TauTask(
        task_id="tau-demo-003",
        user_query="What is your return policy?",
        rules=["Always cite policy document version."],
        expected_actions=[],
        category="retail",
    )

    result_info = TauResult(
        task_id=task_info.task_id,
        model_id="demo-model",
        turns=[
            TauTurn(role="user", content="What is your return policy?"),
            TauTurn(
                role="agent",
                content="Our return policy allows returns within 30 days (Policy v2.1).",
            ),
        ],
        rule_violations=0,
        task_completed=True,
        session_id="tau-session-003",
    )

    return [
        (result_success, task_return),
        (result_failed, task_flight),
        (result_info, task_info),
    ]


def main() -> None:
    print("Recipe 19: tau-bench -> rollout-wire import\n")

    pairs = make_synthetic_tau_data()
    print(f"Synthetic tau-bench result/task pairs: {len(pairs)}")

    # Convert to rollout-wire/v1 format
    rollouts = [tau_to_rollout(result, task) for result, task in pairs]
    print(f"Converted to rollout-wire/v1 records: {len(rollouts)}")
    print()

    for r in rollouts:
        tool_count = sum(1 for e in r["tool_call_sequence"] if e["event"] == "tool_call")
        print(
            f"  rollout_id={r['rollout_id']:<30}  "
            f"status={r['objective_status']:<8}  "
            f"score={r['total_score']:.1f}  "
            f"tool_calls={tool_count}"
        )

    # Also show AEP format for the successful run
    print("\nAEP record for first result (tau-demo-001):")
    aep_first = tau_to_aep(pairs[0][0], pairs[0][1])
    print(f"  run_id         : {aep_first['run_id']}")
    print(f"  actions        : {len(aep_first['actions'])}")
    print(f"  verifier_results: {len(aep_first['verifier_results'])}")
    for v in aep_first["verifier_results"]:
        print(f"    {v['verifier_id']}: passed={v['passed']}  score={v['score']:.2f}")

    # Build AEP records for Evidence Admission scoring
    aep_records = []
    for (result, task), rollout in zip(pairs, rollouts):
        aep_rec = tau_to_aep(result, task)
        # admission_gate expects aep/v0.1 — tau_to_aep already sets this
        aep_records.append(aep_rec)

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
    print("\nRecipe 19 complete.")


if __name__ == "__main__":
    main()
