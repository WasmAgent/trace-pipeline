"""Tests for evomerge.adp.export — rollout-wire/v1 → ADP conversion."""
from __future__ import annotations

from evomerge.adp.export import rollout_to_adp
from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry


def _make_record(n_tools: int = 2) -> RolloutBranchRecord:
    return RolloutBranchRecord(
        rollout_id="rollout-001",
        task="fix the bug",
        branch_index=0,
        temperature=0.7,
        session_id="sess-001",
        tool_call_sequence=[
            ToolCallEntry(
                tool_name=f"tool_{i}",
                arguments={"arg": f"val_{i}"},
                result=f"result_{i}",
            )
            for i in range(n_tools)
        ],
        final_answer="Done.",
        objective_score=1,
        objective_status="pass",
        total_score=100.0,
    )


def test_step_count():
    record = _make_record(n_tools=2)
    steps = rollout_to_adp(record)
    # 2 tools × 2 steps each + 1 terminal = 5
    assert len(steps) == 5


def test_step_indices_sequential():
    steps = rollout_to_adp(_make_record(n_tools=3))
    for i, step in enumerate(steps):
        assert step.step_index == i


def test_agent_environment_interleave():
    steps = rollout_to_adp(_make_record(n_tools=2))
    assert steps[0].role == "agent"
    assert steps[1].role == "environment"
    assert steps[2].role == "agent"
    assert steps[3].role == "environment"


def test_terminal_step():
    steps = rollout_to_adp(_make_record())
    terminal = steps[-1]
    assert terminal.done is True
    assert terminal.role == "agent"
    assert terminal.reward is not None
    assert terminal.content == "Done."


def test_intermediate_steps_not_done():
    steps = rollout_to_adp(_make_record())
    for step in steps[:-1]:
        assert step.done is False
        assert step.reward is None


def test_episode_id():
    steps = rollout_to_adp(_make_record())
    for step in steps:
        assert step.episode_id == "rollout-001/0"


def test_tool_name_on_agent_steps():
    steps = rollout_to_adp(_make_record(n_tools=1))
    agent_step = steps[0]
    assert agent_step.tool_name == "tool_0"
    assert agent_step.tool_args == {"arg": "val_0"}


def test_tool_name_none_on_env_steps():
    steps = rollout_to_adp(_make_record(n_tools=1))
    env_step = steps[1]
    assert env_step.tool_name is None
    assert env_step.tool_args is None


def test_reward_normalized_pass():
    record = _make_record()
    record.objective_score = 1
    record.objective_status = "pass"
    steps = rollout_to_adp(record)
    assert steps[-1].reward is not None
    assert 0.0 <= steps[-1].reward <= 1.0


def test_reward_zero_on_unknown():
    record = _make_record()
    record.objective_status = "unknown"
    steps = rollout_to_adp(record)
    assert steps[-1].reward == 0.0


def test_no_tools_gives_single_step():
    record = _make_record(n_tools=0)
    steps = rollout_to_adp(record)
    assert len(steps) == 1
    assert steps[0].done is True
