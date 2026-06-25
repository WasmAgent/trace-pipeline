"""AEP → router feature bridge.

Extracts RouterFeatures from an AEP record so the router classifier can
operate on evidence from any agent framework (not just compliance runs).

The mapping is approximate — AEP records don't have TaskSpec constraints,
so constraint-related features default to 0. Action and verifier fields
map naturally.
"""
from __future__ import annotations

from typing import Any

from evomerge.router.features import RouterFeatures
from evomerge.router.labels import RouterLabel


def features_from_aep(record: dict[str, Any]) -> RouterFeatures:
    """Extract RouterFeatures from an AEP record dict.

    Mapping:
      actions → eval_tool_calls_total (all actions)
      state_changing actions → eval_tool_calls_valid (successful mutations)
      capability_decisions[deny] → eval_hard_violation_count
      verifier_results[passed=False] → eval_violation_count
      any verifier failed → eval_escalated = 1
    """
    actions = record.get("actions", [])
    cap_decisions = record.get("capability_decisions", [])
    verifier_results = record.get("verifier_results", [])

    tool_calls_total = len(actions)
    state_changing = [a for a in actions if a.get("state_changing")]
    tool_calls_valid = len([a for a in state_changing if not a.get("error")])

    deny_decisions = [d for d in cap_decisions if d.get("decision") == "deny"]
    failed_verifiers = [v for v in verifier_results if not v.get("passed", True)]

    tool_validity_rate = (
        tool_calls_valid / tool_calls_total if tool_calls_total > 0 else 1.0
    )
    escalated = 1 if failed_verifiers else 0

    return RouterFeatures(
        taskspec_n_constraints=0,
        taskspec_n_hard=0,
        taskspec_has_tools=int(tool_calls_total > 0),
        taskspec_n_allowed_tools=0,
        taskspec_max_repair_rounds=0,
        eval_repair_rounds=0,
        eval_violation_count=len(failed_verifiers),
        eval_hard_violation_count=len(deny_decisions),
        eval_tool_calls_total=tool_calls_total,
        eval_tool_calls_valid=tool_calls_valid,
        eval_tool_validity_rate=tool_validity_rate,
        eval_escalated=escalated,
        eval_latency_ms=0.0,
        eval_prompt_tokens=0,
        eval_generation_tokens=0,
    )


def label_from_aep(record: dict[str, Any]) -> RouterLabel:
    """Infer a RouterLabel from AEP verifier results and capability decisions.

    Heuristic:
      - All verifiers passed, no deny decisions → small_model_can_handle
      - Some failed, no deny → need_repair
      - Any deny decision → need_large_model
      - Multiple denies or critical errors → need_human_review
    """
    cap_decisions = record.get("capability_decisions", [])
    verifier_results = record.get("verifier_results", [])

    deny_count = sum(1 for d in cap_decisions if d.get("decision") == "deny")
    failed_count = sum(1 for v in verifier_results if not v.get("passed", True))

    if deny_count >= 2:
        return RouterLabel.need_human_review
    if deny_count == 1:
        return RouterLabel.need_large_model
    if failed_count > 0:
        return RouterLabel.need_repair
    return RouterLabel.small_model_can_handle
