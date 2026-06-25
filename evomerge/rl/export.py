"""RL transition export — convert rollout-wire/v1 to RL-ready transition records.

Produces (state, action, observation, reward, done) tuples compatible with
Agent Lightning / RLVR / GRPO / PPO style training frameworks.

Reward decomposition:
  build   — 1.0 if objective_status=="pass", 0.0 otherwise
  visual  — from build_result metadata (default 0.0 if absent)
  policy  — 1.0 if no firewall_decision=="deny" in step metadata
  cost    — penalty proportional to number of tool calls (0.0 to -0.1)

Usage:
    from evomerge.rl.export import rollout_to_rl_transitions, RlTransition

    transitions = rollout_to_rl_transitions(record, reward_dims=["build","policy"])
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from evomerge.schemas.rollout import RolloutBranchRecord


@dataclass
class RewardSignal:
    build: float = 0.0
    visual: float = 0.0
    policy: float = 1.0
    cost_penalty: float = 0.0

    def total(self) -> float:
        return self.build + self.visual + self.policy + self.cost_penalty


@dataclass
class RlTransition:
    """One (s, a, o, r, done) tuple from a rollout step."""
    episode_id: str
    step_index: int

    # State reference — opaque pointer to pre-action agent state
    state_ref: str
    # Action taken by the agent
    action: dict[str, Any]
    # Observation from the environment
    observation_ref: str

    # Reward (non-None only on terminal step)
    reward: RewardSignal | None
    done: bool

    # Capability tags (from evomerge.capability.taxonomy if available)
    capability_tags: list[str] = field(default_factory=list)

    # Provenance
    rollout_id: str = ""
    branch_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _step_ref(rollout_id: str, branch_index: int, step_index: int, suffix: str) -> str:
    return f"trace://{rollout_id}/{branch_index}/step/{step_index}/{suffix}"


def _action_hash(tool_name: str, arguments: dict) -> str:
    payload = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _compute_reward(
    record: RolloutBranchRecord,
    reward_dims: list[str],
) -> RewardSignal:
    sig = RewardSignal()

    if "build" in reward_dims:
        sig.build = 1.0 if record.objective_status == "pass" else 0.0

    if "visual" in reward_dims:
        # Extract visual score from build_result if present
        br = record.build_result
        if br is not None and hasattr(br, "model_extra"):
            sig.visual = float(br.model_extra.get("visual_score", 0.0))

    if "policy" in reward_dims:
        # 1.0 if no tool call was blocked (no deny in sequence metadata)
        sig.policy = 1.0  # conservative default; real value set by firewall integration

    if "cost" in reward_dims:
        # Penalty: -0.01 per tool call beyond the first, capped at -0.1
        n = len(record.tool_call_sequence)
        sig.cost_penalty = max(-0.1, -0.01 * max(0, n - 1))

    return sig


def rollout_to_rl_transitions(
    record: RolloutBranchRecord,
    *,
    reward_dims: list[str] | None = None,
) -> list[RlTransition]:
    """Convert one RolloutBranchRecord to a list of RL transition records.

    Non-terminal steps have reward=None, done=False.
    The terminal step carries the full RewardSignal, done=True.

    reward_dims controls which reward components are computed.
    Default: ["build", "policy", "cost"].
    """
    if reward_dims is None:
        reward_dims = ["build", "policy", "cost"]

    episode_id = f"{record.rollout_id}/{record.branch_index}"
    transitions: list[RlTransition] = []

    for i, entry in enumerate(record.tool_call_sequence):
        state_ref = _step_ref(record.rollout_id, record.branch_index, i, "pre")
        obs_ref = _step_ref(record.rollout_id, record.branch_index, i, "post")
        action = {
            "tool": entry.tool_name,
            "args_hash": _action_hash(entry.tool_name, entry.arguments),
        }
        transitions.append(RlTransition(
            episode_id=episode_id,
            step_index=i,
            state_ref=state_ref,
            action=action,
            observation_ref=obs_ref,
            reward=None,
            done=False,
            rollout_id=record.rollout_id,
            branch_index=record.branch_index,
            metadata={
                "tool_name": entry.tool_name,
                "has_error": entry.error is not None,
            },
        ))

    # Terminal transition — carries reward
    terminal_idx = len(record.tool_call_sequence)
    reward = _compute_reward(record, reward_dims)
    transitions.append(RlTransition(
        episode_id=episode_id,
        step_index=terminal_idx,
        state_ref=_step_ref(record.rollout_id, record.branch_index, terminal_idx, "pre"),
        action={"type": "final_answer", "args_hash": _action_hash("final_answer", {})},
        observation_ref=_step_ref(record.rollout_id, record.branch_index, terminal_idx, "post"),
        reward=reward,
        done=True,
        rollout_id=record.rollout_id,
        branch_index=record.branch_index,
        metadata={
            "objective_status": record.objective_status,
            "objective_score": record.objective_score,
        },
    ))

    return transitions


def rollout_file_to_rl_transitions(
    rollout_jsonl: str,
    *,
    reward_dims: list[str] | None = None,
    out: str | None = None,
) -> list[RlTransition]:
    """Load a rollout-wire/v1 JSONL file and convert to RL transitions.

    When `out` is set, writes one record per line as JSONL using dataclasses.asdict.
    """
    import dataclasses

    from evomerge.io import load_rollouts

    records = load_rollouts(rollout_jsonl)
    all_transitions: list[RlTransition] = []
    for rec in records:
        all_transitions.extend(rollout_to_rl_transitions(rec, reward_dims=reward_dims))

    if out is not None:
        import pathlib
        path = pathlib.Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for t in all_transitions:
                d = dataclasses.asdict(t)
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    return all_transitions


__all__ = [
    "RewardSignal",
    "RlTransition",
    "rollout_to_rl_transitions",
    "rollout_file_to_rl_transitions",
]
