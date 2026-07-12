"""Training record schemas — matches wasmagent-js training-record JSON Schema.

schema_version tags:
  sft/v1   — supervised fine-tuning record
  dpo/v1   — direct preference optimisation pair
  ppo/v1   — PPO / GRPO reward record
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

LossWeightTokens = Literal["default", "recovery", "state_summary", "high_value"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class Provenance(BaseModel):
    source: str
    rollout_id: str | None = None
    task_id: str | None = None
    n_gram_hash: str | None = None
    task_hash: str | None = None


class TurnAnnotation(BaseModel):
    turn_index: int
    loss_weight_tokens: LossWeightTokens = "default"


class SftTrainingRecord(BaseModel):
    """Full conversation for SFT; the last assistant turn is the training target."""

    schema_version: Literal["sft/v1"] = "sft/v1"
    messages: list[Message]
    output_type: Literal[
        "final_answer", "repair_patch", "tool_call", "next_action", "escalation"
    ]
    loss_weight_tokens: LossWeightTokens = "default"
    loss_weight_annotations: list[TurnAnnotation] | None = None
    provenance: Provenance


class DpoTrainingRecord(BaseModel):
    """DPO preference pair.

    messages: full conversation with chosen as the final assistant turn.
    prompt_messages: messages without the final assistant turn (for TRL).
    chosen / rejected: the two competing assistant responses.
    """

    schema_version: Literal["dpo/v1"] = "dpo/v1"
    messages: list[Message]
    prompt_messages: list[Message] | None = None
    chosen: str
    rejected: str
    loss_weight_tokens: LossWeightTokens = "default"
    loss_weight_annotations: list[TurnAnnotation] | None = None
    provenance: Provenance


class PpoTrainingRecord(BaseModel):
    """PPO / GRPO reward record. reward is normalised to [0, 1]."""

    schema_version: Literal["ppo/v1"] = "ppo/v1"
    messages: list[Message]
    reward: float
    loss_weight_tokens: LossWeightTokens = "default"
    loss_weight_annotations: list[TurnAnnotation] | None = None
    provenance: Provenance


__all__ = [
    "DpoTrainingRecord",
    "LossWeightTokens",
    "Message",
    "PpoTrainingRecord",
    "Provenance",
    "SftTrainingRecord",
    "TurnAnnotation",
]
