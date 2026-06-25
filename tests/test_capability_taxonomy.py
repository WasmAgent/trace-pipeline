"""Tests for evomerge.capability.taxonomy — deterministic capability tagging."""
from __future__ import annotations

from evomerge.capability.taxonomy import (
    CapabilityTag,
    CapabilityTagRecord,
    tag_adp_episode,
    tag_rollout_step,
)


def _step(
    role: str = "agent",
    tool_name: str | None = None,
    content: str = "",
    tool_args: dict | None = None,
    step_index: int = 0,
    metadata: dict | None = None,
) -> dict:
    return {
        "role": role,
        "tool_name": tool_name,
        "content": content,
        "tool_args": tool_args,
        "step_index": step_index,
        "episode_id": "ep-001",
        "metadata": metadata or {},
    }


def test_enum_has_11_members():
    assert len(list(CapabilityTag)) == 11


def test_file_localization_from_read_file():
    tags = tag_rollout_step(_step(tool_name="read_file", step_index=1))
    assert CapabilityTag.FILE_LOCALIZATION in tags


def test_build_repair_from_run_build():
    tags = tag_rollout_step(_step(tool_name="run_build", step_index=1))
    assert CapabilityTag.BUILD_REPAIR in tags


def test_visual_grounding_from_visual():
    tags = tag_rollout_step(_step(tool_name="visual_verify", step_index=1))
    assert CapabilityTag.VISUAL_GROUNDING in tags


def test_insufficient_context():
    tags = tag_rollout_step(_step(
        role="agent",
        tool_name=None,
        content="I need more information to proceed.",
        step_index=0,
    ))
    assert CapabilityTag.INSUFFICIENT_CONTEXT in tags


def test_tool_selection_on_any_tool_call():
    tags = tag_rollout_step(_step(tool_name="some_tool", step_index=1))
    assert CapabilityTag.TOOL_SELECTION in tags


def test_argument_generation_when_args_present():
    tags = tag_rollout_step(_step(
        tool_name="write_file",
        tool_args={"path": "foo.ts", "content": "..."},
        step_index=1,
    ))
    assert CapabilityTag.ARGUMENT_GENERATION in tags


def test_no_argument_generation_when_no_args():
    tags = tag_rollout_step(_step(tool_name="read_file", tool_args=None, step_index=1))
    assert CapabilityTag.ARGUMENT_GENERATION not in tags


def test_recovery_after_error_on_env_error():
    tags = tag_rollout_step(_step(
        role="environment",
        content="TypeError: Cannot read properties of undefined",
        step_index=1,
    ))
    assert CapabilityTag.RECOVERY_AFTER_ERROR in tags


def test_no_duplicates():
    tags = tag_rollout_step(_step(
        tool_name="read_file",
        tool_args={"path": "x"},
        step_index=2,
    ))
    assert len(tags) == len(set(tags))


def test_tags_sorted():
    tags = tag_rollout_step(_step(tool_name="read_file", tool_args={"p": "x"}, step_index=1))
    assert tags == sorted(tags, key=lambda t: t.value)


def test_tag_adp_episode_one_record_per_step():
    steps = [
        _step(tool_name="read_file", step_index=0),
        _step(role="environment", content="file content", step_index=1),
        _step(role="agent", content="Done.", step_index=2),
    ]
    records = tag_adp_episode(steps, task_id="task-001")
    assert len(records) == 3
    for r in records:
        assert isinstance(r, CapabilityTagRecord)
        assert r.task_id == "task-001"
        assert r.confidence == 1.0


def test_tag_adp_episode_empty():
    records = tag_adp_episode([], task_id="empty")
    assert records == []
