"""Convert RolloutBranchRecord list → SFT training records.

One SFT record per branch (all branches, not just winners). The training target
is the final_answer; the conversation prefix reconstructs the tool call
sequence so the model sees exactly what the agent saw.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry
from evomerge.schemas.training import Message, Provenance, SftTrainingRecord


_PRIORITY = {"high_value": 3, "recovery": 2, "state_summary": 1, "default": 0}


def _collapse_annotations(annotations: list | None) -> str:
    """Pick highest-priority label from per-turn annotations."""
    if not annotations:
        return "default"
    best = "default"
    for ann in annotations:
        label = (
            ann.get("loss_weight_tokens", "default")
            if isinstance(ann, dict)
            else getattr(ann, "loss_weight_tokens", "default")
        )
        if _PRIORITY.get(label, 0) > _PRIORITY.get(best, 0):
            best = label
    return best


def _normalize_tool_sequence(raw_sequence: list) -> list[ToolCallEntry]:
    """Convert upstream AgentEvent[] to ToolCallEntry[] for message building.

    Handles both the flat ToolCallEntry format and the nested AgentEvent format.
    """
    entries: list[ToolCallEntry] = []

    if not raw_sequence:
        return entries

    # Check if already in ToolCallEntry format
    if isinstance(raw_sequence[0], ToolCallEntry):
        return raw_sequence
    if hasattr(raw_sequence[0], "tool_name"):
        return raw_sequence

    # Convert AgentEvent pairs to ToolCallEntry
    calls: dict[str, dict] = {}  # callId -> tool_call data
    results: dict[str, dict] = {}  # callId -> tool_result data

    for item in raw_sequence:
        if isinstance(item, dict):
            event_type = item.get("event", "")
            data = item.get("data", {})
        else:
            event_type = getattr(item, "event", "")
            data = getattr(item, "data", {})

        if event_type == "tool_call":
            call_id = data.get("callId", "")
            calls[call_id] = data
        elif event_type == "tool_result":
            call_id = data.get("callId", "")
            results[call_id] = data

    # Pair calls with results by callId
    for call_id, call_data in calls.items():
        result_data = results.get(call_id, {})
        tool_name = call_data.get("toolName", "")
        arguments = call_data.get("args", {})
        output = result_data.get("output")
        error = result_data.get("error")
        error_str = None
        if error:
            if isinstance(error, dict):
                error_str = error.get("message", str(error))
            elif isinstance(error, str):
                error_str = error
        entries.append(
            ToolCallEntry(
                tool_name=tool_name,
                arguments=arguments,
                result=output,
                error=error_str,
            )
        )

    return entries


def _task_hash(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()[:16]


def _build_messages(record: RolloutBranchRecord) -> list[Message]:
    msgs: list[Message] = [Message(role="user", content=record.task)]
    normalized = _normalize_tool_sequence(record.tool_call_sequence)
    for entry in normalized:
        if entry.tool_name:
            msgs.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"name": entry.tool_name, "arguments": entry.arguments}
                    ],
                )
            )
            result_content = (
                str(entry.result) if entry.result is not None else (entry.error or "")
            )
            msgs.append(
                Message(role="tool", content=result_content)
            )
    msgs.append(Message(role="assistant", content=record.final_answer))
    return msgs


def to_sft_records(
    rollouts: Sequence[RolloutBranchRecord],
    *,
    only_passing: bool = True,
) -> list[SftTrainingRecord]:
    """Convert rollout branches to SFT records.

    Args:
        rollouts: branches from RolloutForkRunner.
        only_passing: when True (default) only include branches with
            objective_score == 1.  Set False to include all branches.

    Returns:
        List of SftTrainingRecord, one per qualifying branch.
    """
    records: list[SftTrainingRecord] = []
    for r in rollouts:
        if only_passing and r.objective_score != 1:
            continue
        prov = Provenance(
            source="wasmagent-rollout",
            rollout_id=r.rollout_id,
            task_hash=_task_hash(r.task),
        )
        records.append(
            SftTrainingRecord(
                messages=_build_messages(r),
                output_type="final_answer",
                provenance=prov,
            )
        )
    return records
