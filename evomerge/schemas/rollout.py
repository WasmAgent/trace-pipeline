"""Rollout wire format — matches wasmagent-js rollout-wire/v1 JSON Schema."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCallEntry(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class ToolCallData(BaseModel, extra="allow"):
    toolName: str = Field(alias="toolName")
    args: dict[str, Any] = Field(default_factory=dict)
    callId: str = ""


class ToolResultData(BaseModel, extra="allow"):
    callId: str = ""
    toolName: str = Field(default="", alias="toolName")
    output: Any = None
    error: Any = None


class AgentEvent(BaseModel, extra="allow"):
    channel: str = "tool"
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class BuildResult(BaseModel):
    status: Literal["pass", "fail", "skip"]
    exit_code: int | None = None
    stderr: str | None = None


class RolloutBranchRecord(BaseModel):
    """One branch produced by wasmagent-js RolloutForkRunner.

    schema_version is always "rollout-wire/v1" to match the SSOT in
    wasmagent-js/packages/core/src/ranking/schemas/rollout-wire.schema.json.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["rollout-wire/v1"] = "rollout-wire/v1"
    rollout_id: str
    task: str
    branch_index: int
    temperature: float
    session_id: str
    tool_call_sequence: list[ToolCallEntry | dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    build_result: BuildResult | None = None
    seed: int | None = None
    # RolloutRanker-enriched fields
    objective_score: Literal[0, 1] = 0
    objective_status: Literal["pass", "fail", "unknown"] = "unknown"
    rank: int = 0
    total_score: float = 0.0


class ForkBranch(BaseModel):
    branch_index: int
    forked_at_step: int
    forked_at_event_id: str = ""
    shared_prefix_steps: int = 0


class RolloutTreeRecord(BaseModel):
    """Fork-point topology for step-level DPO credit assignment.

    Emitted by wasmagent-js RolloutTreeExporter alongside RolloutBranchRecords.
    Currently used for informational purposes only — downstream pairing uses
    the flat branch records.
    """

    schema_version: Literal["rollout-wire/v1"] = "rollout-wire/v1"
    rollout_id: str
    task: str
    branches: list[ForkBranch] = Field(default_factory=list)
    fork_map: dict[str, list[int]] = Field(default_factory=dict)


__all__ = [
    "AgentEvent",
    "BuildResult",
    "ForkBranch",
    "RolloutBranchRecord",
    "RolloutTreeRecord",
    "ToolCallData",
    "ToolCallEntry",
    "ToolResultData",
]
