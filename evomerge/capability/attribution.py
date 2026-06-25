"""Capability attribution engine — mine capability gaps from rollout traces.

Analyzes successful and failed rollout branches to identify:
  - Which capability tags are associated with success vs failure
  - Which step patterns cause failures (failure_class)
  - What training records to generate (DPO pairs, SFT targeted records)

This is the evomerge side of TRACE/Agent-RLVR-style capability targeting:
not generic SFT, but targeted training on specific failure modes.

Usage:
    from evomerge.capability.attribution import (
        attribute_rollout, AttributionResult, mine_capability_gaps
    )

    result = attribute_rollout(successful_record, failed_record)
    gaps = mine_capability_gaps(rollout_list)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from evomerge.adp.export import rollout_to_adp
from evomerge.capability.taxonomy import (
    CapabilityTag,
    CapabilityTagRecord,
    tag_adp_episode,
)
from evomerge.schemas.rollout import RolloutBranchRecord


# ── Failure classes ──────────────────────────────────────────────────────────

FAILURE_CLASSES = [
    "planning",
    "file_localization",
    "tool_selection",
    "argument_generation",
    "state_tracking",
    "insufficient_context_detection",
    "recovery_after_error",
    "policy_compliance",
    "cost_control",
    "visual_grounding",
    "build_repair",
    "unknown",
]


@dataclass
class CapabilityGap:
    """A specific capability that a failed rollout lacked relative to a successful one."""
    task_id: str
    failure_class: str
    evidence: list[str]
    """Short excerpts from failed steps that triggered the classification."""
    suggested_training_type: str
    """'dpo_pair' | 'sft_targeted' | 'rl_transition'"""
    positive_step_ref: str | None
    """Reference to a successful step demonstrating the capability."""
    negative_step_ref: str | None
    """Reference to the failing step."""
    reward_signal: dict[str, float] = field(default_factory=dict)
    """build / policy / cost signals from the failed branch."""


@dataclass
class AttributionResult:
    """Attribution of a failed rollout relative to a passing one."""
    rollout_id: str
    passing_branch: int
    failing_branch: int
    gaps: list[CapabilityGap]
    passing_tags: list[str]
    failing_tags: list[str]
    tag_diff: list[str]
    """Tags present in passing but absent in failing branch."""


@dataclass
class MineResult:
    """Aggregate capability gap analysis over a list of rollout branches."""
    n_rollouts: int
    n_compared_pairs: int
    gaps: list[CapabilityGap]
    failure_class_counts: dict[str, int]
    suggested_dpo_pairs: int
    suggested_sft_targeted: int


# ── Core attribution logic ───────────────────────────────────────────────────

def _tag_set(records: list[CapabilityTagRecord]) -> set[str]:
    tags: set[str] = set()
    for r in records:
        for t in r.tags:
            tags.add(t.value)
    return tags


def _classify_failure(
    failing_records: list[CapabilityTagRecord],
    passing_tags: set[str],
    failing_tags: set[str],
) -> str:
    """Heuristic: classify the dominant failure mode."""
    diff = passing_tags - failing_tags

    # Priority ordering mirrors TRACE paper's capability taxonomy
    priority = [
        (CapabilityTag.FILE_LOCALIZATION, "file_localization"),
        (CapabilityTag.ARGUMENT_GENERATION, "argument_generation"),
        (CapabilityTag.PLANNING, "planning"),
        (CapabilityTag.RECOVERY_AFTER_ERROR, "recovery_after_error"),
        (CapabilityTag.STATE_TRACKING, "state_tracking"),
        (CapabilityTag.BUILD_REPAIR, "build_repair"),
        (CapabilityTag.VISUAL_GROUNDING, "visual_grounding"),
        (CapabilityTag.POLICY_COMPLIANCE, "policy_compliance"),
        (CapabilityTag.COST_CONTROL, "cost_control"),
        (CapabilityTag.INSUFFICIENT_CONTEXT, "insufficient_context_detection"),
    ]
    for tag, name in priority:
        if tag.value in diff:
            return name
    return "unknown"


def attribute_rollout(
    passing: RolloutBranchRecord,
    failing: RolloutBranchRecord,
) -> AttributionResult:
    """Compare a passing and failing branch from the same rollout.

    Returns which capability tags the passing branch had that the failing
    branch lacked, and suggests training records to close the gap.
    """
    pass_steps = rollout_to_adp(passing)
    fail_steps = rollout_to_adp(failing)

    pass_tags_records = tag_adp_episode(
        [dataclasses.asdict(s) for s in pass_steps], task_id=passing.rollout_id
    )
    fail_tags_records = tag_adp_episode(
        [dataclasses.asdict(s) for s in fail_steps], task_id=failing.rollout_id
    )

    passing_tags = _tag_set(pass_tags_records)
    failing_tags = _tag_set(fail_tags_records)
    tag_diff = sorted(passing_tags - failing_tags)

    gaps: list[CapabilityGap] = []
    if tag_diff:
        failure_class = _classify_failure(fail_tags_records, passing_tags, failing_tags)

        # Find evidence: steps in failing branch that relate to the gap
        evidence: list[str] = []
        pos_ref: str | None = None
        neg_ref: str | None = None

        for r in fail_tags_records:
            if any(t.value in tag_diff for t in r.tags):
                evidence.append(r.evidence_snippet[:80])
                neg_ref = r.step_ref
                break

        for r in pass_tags_records:
            if any(t.value in tag_diff for t in r.tags):
                pos_ref = r.step_ref
                break

        reward = {
            "build": float(failing.objective_score),
            "policy": 1.0,
            "cost": max(-0.1, -0.01 * max(0, len(failing.tool_call_sequence) - 1)),
        }

        gaps.append(CapabilityGap(
            task_id=failing.rollout_id,
            failure_class=failure_class,
            evidence=evidence,
            suggested_training_type="dpo_pair" if failing.objective_status in ("pass", "fail") else "sft_targeted",
            positive_step_ref=pos_ref,
            negative_step_ref=neg_ref,
            reward_signal=reward,
        ))

    return AttributionResult(
        rollout_id=passing.rollout_id,
        passing_branch=passing.branch_index,
        failing_branch=failing.branch_index,
        gaps=gaps,
        passing_tags=sorted(passing_tags),
        failing_tags=sorted(failing_tags),
        tag_diff=tag_diff,
    )


def mine_capability_gaps(
    rollouts: list[RolloutBranchRecord],
) -> MineResult:
    """Mine capability gaps across a list of rollout branches.

    Groups branches by rollout_id, pairs each failing branch (objective_score=0)
    against the best passing branch (objective_score=1) from the same rollout.
    Returns aggregate gap analysis.
    """
    from collections import defaultdict

    by_rollout: dict[str, list[RolloutBranchRecord]] = defaultdict(list)
    for r in rollouts:
        by_rollout[r.rollout_id].append(r)

    all_gaps: list[CapabilityGap] = []
    n_pairs = 0
    failure_counts: dict[str, int] = {}

    for rollout_id, branches in by_rollout.items():
        passing = [b for b in branches if b.objective_score == 1]
        failing = [b for b in branches if b.objective_score == 0]
        if not passing or not failing:
            continue
        # Use the highest-ranked passing branch
        best_pass = max(passing, key=lambda b: b.total_score or 0)
        for fail_branch in failing:
            attr = attribute_rollout(best_pass, fail_branch)
            all_gaps.extend(attr.gaps)
            n_pairs += 1
            for gap in attr.gaps:
                failure_counts[gap.failure_class] = failure_counts.get(gap.failure_class, 0) + 1

    dpo = sum(1 for g in all_gaps if g.suggested_training_type == "dpo_pair")
    sft = sum(1 for g in all_gaps if g.suggested_training_type == "sft_targeted")

    return MineResult(
        n_rollouts=len(by_rollout),
        n_compared_pairs=n_pairs,
        gaps=all_gaps,
        failure_class_counts=failure_counts,
        suggested_dpo_pairs=dpo,
        suggested_sft_targeted=sft,
    )


__all__ = [
    "CapabilityGap",
    "AttributionResult",
    "MineResult",
    "FAILURE_CLASSES",
    "attribute_rollout",
    "mine_capability_gaps",
]
