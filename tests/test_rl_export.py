"""Tests for evomerge.rl.export — RL transition records."""
from __future__ import annotations

from evomerge.rl.export import RewardSignal, rollout_to_rl_transitions
from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry


def _make_record(n_tools: int = 2, status: str = "pass") -> RolloutBranchRecord:
    return RolloutBranchRecord(
        rollout_id="rollout-001",
        task="fix the bug",
        branch_index=0,
        temperature=0.7,
        session_id="sess-001",
        tool_call_sequence=[
            ToolCallEntry(tool_name=f"tool_{i}", arguments={"k": f"v_{i}"}, result="ok")
            for i in range(n_tools)
        ],
        final_answer="Done.",
        objective_score=1 if status == "pass" else 0,
        objective_status=status,
        total_score=100.0,
    )


def test_transition_count():
    # n tools + 1 terminal
    transitions = rollout_to_rl_transitions(_make_record(n_tools=3))
    assert len(transitions) == 4


def test_step_indices_sequential():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=2))
    for i, t in enumerate(transitions):
        assert t.step_index == i


def test_terminal_is_last():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=2))
    assert transitions[-1].done is True
    assert transitions[-1].reward is not None


def test_intermediate_no_reward():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=2))
    for t in transitions[:-1]:
        assert t.done is False
        assert t.reward is None


def test_build_reward_pass():
    transitions = rollout_to_rl_transitions(
        _make_record(status="pass"), reward_dims=["build"]
    )
    assert transitions[-1].reward.build == 1.0


def test_build_reward_fail():
    transitions = rollout_to_rl_transitions(
        _make_record(status="fail"), reward_dims=["build"]
    )
    assert transitions[-1].reward.build == 0.0


def test_cost_penalty():
    # 3 tools → penalty = -0.02 (2 extra calls × -0.01)
    transitions = rollout_to_rl_transitions(
        _make_record(n_tools=3), reward_dims=["cost"]
    )
    assert transitions[-1].reward.cost_penalty == pytest.approx(-0.02)


def test_cost_penalty_capped():
    # 20 tools → penalty capped at -0.1
    transitions = rollout_to_rl_transitions(
        _make_record(n_tools=20), reward_dims=["cost"]
    )
    assert transitions[-1].reward.cost_penalty == pytest.approx(-0.1)


def test_episode_id():
    transitions = rollout_to_rl_transitions(_make_record())
    for t in transitions:
        assert t.episode_id == "rollout-001/0"


def test_state_ref_format():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=1))
    assert transitions[0].state_ref.startswith("trace://rollout-001/0/step/0/pre")


def test_action_contains_tool_name():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=1))
    assert transitions[0].action["tool"] == "tool_0"


def test_total_reward():
    sig = RewardSignal(build=1.0, policy=1.0, cost_penalty=-0.02)
    assert sig.total() == pytest.approx(1.98)


def test_no_tools_gives_single_transition():
    transitions = rollout_to_rl_transitions(_make_record(n_tools=0))
    assert len(transitions) == 1
    assert transitions[0].done is True


import pytest
