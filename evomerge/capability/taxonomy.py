"""Capability tag taxonomy for agent rollout steps.

Deterministic, rule-based — no ML required. Operates on plain dicts
so it works with both ADP steps and raw rollout tool_call_sequence entries.

Usage:
    from evomerge.capability.taxonomy import (
        CapabilityTag, CapabilityTagRecord,
        tag_rollout_step, tag_adp_episode,
    )

    tags = tag_rollout_step({"tool_name": "read_file", "content": "", "role": "agent"})
    records = tag_adp_episode(steps, task_id="task-001")
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CapabilityTag(str, Enum):
    PLANNING                 = "planning"
    FILE_LOCALIZATION        = "file_localization"
    TOOL_SELECTION           = "tool_selection"
    ARGUMENT_GENERATION      = "argument_generation"
    STATE_TRACKING           = "state_tracking"
    INSUFFICIENT_CONTEXT     = "insufficient_context_detection"
    RECOVERY_AFTER_ERROR     = "recovery_after_error"
    POLICY_COMPLIANCE        = "policy_compliance"
    COST_CONTROL             = "cost_control"
    VISUAL_GROUNDING         = "visual_grounding"
    BUILD_REPAIR             = "build_repair"


@dataclass
class CapabilityTagRecord:
    task_id: str
    step_ref: str           # e.g. "rollout_id/branch_index/step_index"
    tags: list[CapabilityTag]
    confidence: float       # 0.0–1.0; deterministic rules always emit 1.0
    evidence_snippet: str   # short excerpt that triggered the tag(s)


# ---------------------------------------------------------------------------
# Rule tables — module-level so callers can monkey-patch for tests
# ---------------------------------------------------------------------------

# Maps tool_name substrings → CapabilityTag (checked via `substring in tool_name.lower()`)
TOOL_NAME_RULES: dict[str, CapabilityTag] = {
    "read_file":    CapabilityTag.FILE_LOCALIZATION,
    "write_file":   CapabilityTag.FILE_LOCALIZATION,
    "search":       CapabilityTag.FILE_LOCALIZATION,
    "visual":       CapabilityTag.VISUAL_GROUNDING,
    "screenshot":   CapabilityTag.VISUAL_GROUNDING,
    "build":        CapabilityTag.BUILD_REPAIR,
    "run_build":    CapabilityTag.BUILD_REPAIR,
    "run_test":     CapabilityTag.BUILD_REPAIR,
    "typecheck":    CapabilityTag.BUILD_REPAIR,
    "shell":        CapabilityTag.COST_CONTROL,
    "approve":      CapabilityTag.POLICY_COMPLIANCE,
    "policy":       CapabilityTag.POLICY_COMPLIANCE,
}

# Maps verifier_result dict keys → CapabilityTag
VERIFIER_RULES: dict[str, CapabilityTag] = {
    "build_passes":   CapabilityTag.BUILD_REPAIR,
    "test_passes":    CapabilityTag.BUILD_REPAIR,
    "policy_ok":      CapabilityTag.POLICY_COMPLIANCE,
    "visual_assert":  CapabilityTag.VISUAL_GROUNDING,
}

_INSUFFICIENT_SIGNALS = ("insufficient", "more information", "need more", "unclear", "ambiguous")
_ERROR_SIGNALS = ("error", "traceback", "exception", "Error:", "Traceback")
_PLANNING_SIGNALS = ("plan", "step 1", "first,", "i will", "let me")


def tag_rollout_step(step: dict[str, Any]) -> list[CapabilityTag]:
    """Deterministic rule-based tagger for a single step dict.

    Expected keys (all optional but commonly present):
      tool_name (str|None), content (str), role (str),
      tool_args (dict|None), metadata (dict, optional).

    Returns a sorted, deduplicated list of CapabilityTag. Never raises.
    """
    tags: set[CapabilityTag] = set()
    tool_name: str | None = step.get("tool_name")
    content: str = step.get("content") or ""
    role: str = step.get("role") or ""
    tool_args: dict | None = step.get("tool_args")
    metadata: dict = step.get("metadata") or {}

    # --- tool_name rules ---
    if tool_name:
        tool_lower = tool_name.lower()
        for substring, tag in TOOL_NAME_RULES.items():
            if substring in tool_lower:
                tags.add(tag)
        # any tool call implies a selection was made
        tags.add(CapabilityTag.TOOL_SELECTION)

    # --- argument_generation ---
    if tool_args and isinstance(tool_args, dict) and len(tool_args) > 0:
        tags.add(CapabilityTag.ARGUMENT_GENERATION)

    # --- planning (agent, content contains planning signals) ---
    if role == "agent" and tool_name is None:
        content_lower = content.lower()
        if any(sig in content_lower for sig in _PLANNING_SIGNALS):
            tags.add(CapabilityTag.PLANNING)

    # --- insufficient_context_detection ---
    if role == "agent" and tool_name is None:
        content_lower = content.lower()
        if any(sig in content_lower for sig in _INSUFFICIENT_SIGNALS):
            tags.add(CapabilityTag.INSUFFICIENT_CONTEXT)

    # --- recovery_after_error (environment step with error content) ---
    if role == "environment":
        if any(sig in content for sig in _ERROR_SIGNALS):
            tags.add(CapabilityTag.RECOVERY_AFTER_ERROR)

    # --- state_tracking (agent following an environment step) ---
    if role == "agent" and step.get("step_index", 0) > 0:
        tags.add(CapabilityTag.STATE_TRACKING)

    # --- verifier_result metadata ---
    verifier_result: dict = metadata.get("verifier_result") or {}
    for key, tag in VERIFIER_RULES.items():
        if key in verifier_result:
            tags.add(tag)

    return sorted(tags, key=lambda t: t.value)


def tag_adp_episode(
    steps: list[dict[str, Any]],
    task_id: str,
) -> list[CapabilityTagRecord]:
    """Tag every step in an ADP episode and return one CapabilityTagRecord per step."""
    records: list[CapabilityTagRecord] = []
    for step in steps:
        tags = tag_rollout_step(step)
        step_index = step.get("step_index", len(records))
        episode_id = step.get("episode_id", task_id)
        step_ref = f"{episode_id}/{step_index}"
        snippet = (step.get("tool_name") or step.get("content") or "")[:120]
        records.append(CapabilityTagRecord(
            task_id=task_id,
            step_ref=step_ref,
            tags=tags,
            confidence=1.0,
            evidence_snippet=snippet,
        ))
    return records


__all__ = [
    "CapabilityTag",
    "CapabilityTagRecord",
    "TOOL_NAME_RULES",
    "VERIFIER_RULES",
    "tag_rollout_step",
    "tag_adp_episode",
]
