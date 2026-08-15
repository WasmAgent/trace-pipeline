"""Counterfactual repair trajectory synthesis.

Transforms AEP records containing policy violations or execution failures
into DPO training pairs, where:
  - rejected = the actual (failing/violating) action narrative
  - chosen   = the repair / safe-refusal narrative

These negative target pairs are used for DPO dataset generation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from evomerge.schemas.training import DpoTrainingRecord, Message, Provenance


def _action_narrative(action: dict[str, Any]) -> str:
    tool = action.get("tool_name", "unknown")
    action_id = action.get("action_id", "")
    return f"[tool={tool} action_id={action_id}]"


def _task_hash(record: dict[str, Any], action_id: str) -> str:
    payload = json.dumps({"run_id": record.get("run_id", ""), "action_id": action_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def synthesize_from_policy_violations(
    record: dict[str, Any],
) -> list[DpoTrainingRecord]:
    """Return DPO pairs for each capability_decision with decision='deny'."""
    pairs: list[DpoTrainingRecord] = []
    run_id = record.get("run_id", "")
    denied = [d for d in record.get("capability_decisions", []) if d.get("decision") == "deny"]
    if not denied:
        return pairs

    for decision in denied:
        capability = decision.get("capability", "")
        resource = decision.get("resource", "")
        reason_code = decision.get("reason_code", "")
        tool = decision.get("subject", capability)
        reason = reason_code or f"capability '{capability}' is denied for resource '{resource}'"

        system_msg = Message(role="system", content="You are an AI agent operating under strict capability policies.")
        user_msg = Message(role="user", content=f"Perform action using tool `{tool}` on resource `{resource}`.")
        rejected_msg = f"I will invoke `{tool}` on `{resource}`."
        chosen_msg = (
            f"I cannot invoke `{tool}` on `{resource}` because {reason}. "
            "I will report this to the user and stop."
        )

        prov = Provenance(
            source="counterfactual/policy-violation",
            rollout_id=run_id,
            task_hash=_task_hash(record, f"cap:{capability}:{resource}"),
        )
        pairs.append(DpoTrainingRecord(
            messages=[system_msg, user_msg, Message(role="assistant", content=chosen_msg)],
            prompt_messages=[system_msg, user_msg],
            chosen=chosen_msg,
            rejected=rejected_msg,
            loss_weight_tokens="recovery",
            provenance=prov,
        ))
    return pairs


def synthesize_from_verifier_failures(
    record: dict[str, Any],
) -> list[DpoTrainingRecord]:
    """Return DPO pairs for each verifier_result with passed=False."""
    pairs: list[DpoTrainingRecord] = []
    run_id = record.get("run_id", "")
    failed = [v for v in record.get("verifier_results", []) if not v.get("passed", True)]
    if not failed:
        return pairs

    sc_actions = [a for a in record.get("actions", []) if a.get("state_changing")]
    context_action = sc_actions[-1] if sc_actions else {}
    action_narrative = _action_narrative(context_action) if context_action else "[no action]"

    for vr in failed:
        verifier_id = vr.get("verifier_id", "")
        score = vr.get("score", 0.0)
        reason = f"verifier '{verifier_id}' failed (score={score:.2f})"

        system_msg = Message(role="system", content="You are an AI agent. Your actions are verified for correctness and safety.")
        user_msg = Message(role="user", content=f"Complete the task. Last action: {action_narrative}")
        rejected_msg = f"I performed {action_narrative} and proceeded, but verification failed: {reason}."
        chosen_msg = (
            f"The action {action_narrative} did not pass verification ({reason}). "
            "I will pause, review the failure, and either retry with a corrected approach or escalate to the user."
        )

        prov = Provenance(
            source="counterfactual/verifier-failure",
            rollout_id=run_id,
            task_hash=_task_hash(record, f"ver:{verifier_id}"),
        )
        pairs.append(DpoTrainingRecord(
            messages=[system_msg, user_msg, Message(role="assistant", content=chosen_msg)],
            prompt_messages=[system_msg, user_msg],
            chosen=chosen_msg,
            rejected=rejected_msg,
            loss_weight_tokens="recovery",
            provenance=prov,
        ))
    return pairs


def synthesize_counterfactuals(
    records: list[dict[str, Any]],
) -> list[DpoTrainingRecord]:
    """Transform AEP records into DPO training pairs.

    Processes both policy violations and verifier failures. Records with
    neither are skipped.
    """
    result: list[DpoTrainingRecord] = []
    for record in records:
        result.extend(synthesize_from_policy_violations(record))
        result.extend(synthesize_from_verifier_failures(record))
    return result


__all__ = [
    "synthesize_counterfactuals",
    "synthesize_from_policy_violations",
    "synthesize_from_verifier_failures",
]
