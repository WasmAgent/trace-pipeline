"""Convert RolloutBranchRecord list → PPO / GRPO reward records.

Reward is taken directly from objective_score (0 or 1) for simplicity.
Callers can replace the reward function via the reward_fn parameter.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from evomerge.pipeline.sft import _build_messages, _task_hash
from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import PpoTrainingRecord, Provenance


def to_ppo_records(
    rollouts: Sequence[RolloutBranchRecord],
    *,
    reward_fn: Callable[[RolloutBranchRecord], float] | None = None,
) -> list[PpoTrainingRecord]:
    """Convert every branch to a PPO reward record.

    Args:
        rollouts: all branches (pass/fail alike).
        reward_fn: optional override; defaults to float(objective_score).

    Returns:
        List of PpoTrainingRecord, one per branch.
    """
    records: list[PpoTrainingRecord] = []
    for r in rollouts:
        reward = reward_fn(r) if reward_fn is not None else float(r.objective_score)
        prov = Provenance(
            source="wasmagent-rollout",
            rollout_id=r.rollout_id,
            task_hash=_task_hash(r.task),
        )
        records.append(
            PpoTrainingRecord(
                messages=_build_messages(r),
                reward=reward,
                provenance=prov,
            )
        )
    return records
