"""ADP (Agent Data Protocol) export — convert rollout-wire/v1 to ADP episodes.

Reference: Agent Data Protocol, arXiv:2510.24702.

Each RolloutBranchRecord becomes one ADP episode:
  - Interleaved agent/environment steps from tool_call_sequence
  - Final agent step carrying final_answer (done=True, reward=normalized score)

Usage:
    from evomerge.adp.export import rollout_to_adp, rollout_file_to_adp

    records = rollout_file_to_adp("data/rollouts.jsonl", out="data/adp.jsonl")
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evomerge.io import load_rollouts
from evomerge.schemas.rollout import RolloutBranchRecord


@dataclass
class AdpRecord:
    episode_id: str
    step_index: int
    role: str            # "agent" | "environment"
    content: str
    tool_name: str | None
    tool_args: dict | None
    reward: float | None
    done: bool
    metadata: dict = field(default_factory=dict)


def _normalize_reward(record: RolloutBranchRecord) -> float:
    """Map objective_score + total_score to [0, 1] float reward."""
    if record.objective_status == "unknown":
        return 0.0
    base = float(record.objective_score)
    bonus = max(0.0, min(0.1, (record.total_score or 0.0) / 1000.0))
    return min(1.0, base + bonus)


def rollout_to_adp(record: RolloutBranchRecord) -> list[AdpRecord]:
    """Convert one RolloutBranchRecord to a flat list of ADP steps.

    Step layout:
      For each ToolCallEntry at index i:
        step 2i   — role="agent",       tool call
        step 2i+1 — role="environment", observation
      Final step  — role="agent",       final_answer, done=True, reward=...
    """
    episode_id = f"{record.rollout_id}/{record.branch_index}"
    base_meta = {
        "rollout_id": record.rollout_id,
        "branch_index": record.branch_index,
        "objective_score": record.objective_score,
        "objective_status": record.objective_status,
        "temperature": record.temperature,
        "session_id": record.session_id,
    }
    steps: list[AdpRecord] = []

    for i, entry in enumerate(record.tool_call_sequence):
        # agent step — tool call
        steps.append(AdpRecord(
            episode_id=episode_id,
            step_index=len(steps),
            role="agent",
            content=json.dumps({"name": entry.tool_name, "arguments": entry.arguments},
                               ensure_ascii=False),
            tool_name=entry.tool_name,
            tool_args=entry.arguments,
            reward=None,
            done=False,
            metadata={**base_meta, "tool_call_index": i},
        ))
        # environment step — observation / result
        obs = entry.result if entry.result is not None else (entry.error or "")
        steps.append(AdpRecord(
            episode_id=episode_id,
            step_index=len(steps),
            role="environment",
            content=str(obs) if not isinstance(obs, str) else obs,
            tool_name=None,
            tool_args=None,
            reward=None,
            done=False,
            metadata={**base_meta, "tool_call_index": i, "is_error": entry.error is not None},
        ))

    # terminal agent step
    steps.append(AdpRecord(
        episode_id=episode_id,
        step_index=len(steps),
        role="agent",
        content=record.final_answer,
        tool_name=None,
        tool_args=None,
        reward=_normalize_reward(record),
        done=True,
        metadata={**base_meta, "is_terminal": True},
    ))
    return steps


def rollout_file_to_adp(
    rollout_jsonl: str | Path,
    *,
    out: str | Path | None = None,
) -> list[AdpRecord]:
    """Load a rollout-wire/v1 JSONL file and convert all records to ADP.

    When `out` is set, writes one AdpRecord per line as JSONL.
    Always returns the full list of records.
    """
    records = load_rollouts(str(rollout_jsonl))
    all_steps: list[AdpRecord] = []
    for record in records:
        all_steps.extend(rollout_to_adp(record))

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for step in all_steps:
                fh.write(json.dumps(dataclasses.asdict(step), ensure_ascii=False) + "\n")

    return all_steps


__all__ = ["AdpRecord", "rollout_to_adp", "rollout_file_to_adp"]
