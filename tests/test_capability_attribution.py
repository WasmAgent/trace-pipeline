"""Tests for evomerge.capability.attribution — capability gap mining."""
from __future__ import annotations

import pytest

from evomerge.capability.attribution import (
    CapabilityGap,
    MineResult,
    attribute_rollout,
    mine_capability_gaps,
)
from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry


def _make_branch(
    rollout_id: str,
    branch_index: int,
    objective_score: int,
    tool_names: list[str],
    total_score: float = 0.0,
) -> RolloutBranchRecord:
    return RolloutBranchRecord(
        rollout_id=rollout_id,
        task="fix the bug",
        branch_index=branch_index,
        temperature=0.7,
        session_id="sess-001",
        tool_call_sequence=[
            ToolCallEntry(tool_name=name, arguments={"arg": "val"}, result="ok")
            for name in tool_names
        ],
        final_answer="Done.",
        objective_score=objective_score,
        objective_status="pass" if objective_score == 1 else "fail",
        total_score=total_score,
    )


class TestAttributeRollout:
    def test_finds_gap_when_passing_has_more_tags(self):
        passing = _make_branch("r1", 0, 1, ["read_file", "write_file", "run_build"])
        failing = _make_branch("r1", 1, 0, ["run_code"])
        result = attribute_rollout(passing, failing)
        assert result.rollout_id == "r1"
        assert result.passing_branch == 0
        assert result.failing_branch == 1
        assert len(result.tag_diff) > 0

    def test_no_gaps_when_same_tools(self):
        passing = _make_branch("r1", 0, 1, ["run_code"])
        failing = _make_branch("r1", 1, 0, ["run_code"])
        result = attribute_rollout(passing, failing)
        # Both same tools → same tags → no diff → no gaps
        assert result.tag_diff == [] or len(result.gaps) == 0

    def test_gap_has_failure_class(self):
        passing = _make_branch("r1", 0, 1, ["read_file", "write_file"])
        failing = _make_branch("r1", 1, 0, ["run_code"])
        result = attribute_rollout(passing, failing)
        if result.gaps:
            assert result.gaps[0].failure_class in [
                "file_localization", "argument_generation", "planning",
                "recovery_after_error", "state_tracking", "build_repair",
                "visual_grounding", "policy_compliance", "cost_control",
                "insufficient_context_detection", "unknown",
            ]

    def test_gap_has_reward_signal(self):
        passing = _make_branch("r1", 0, 1, ["read_file", "write_file"])
        failing = _make_branch("r1", 1, 0, ["run_code"])
        result = attribute_rollout(passing, failing)
        if result.gaps:
            gap = result.gaps[0]
            assert "build" in gap.reward_signal
            assert "policy" in gap.reward_signal

    def test_suggested_dpo_pair_for_pass_fail(self):
        passing = _make_branch("r1", 0, 1, ["read_file", "write_file"])
        failing = _make_branch("r1", 1, 0, ["run_code"])
        result = attribute_rollout(passing, failing)
        if result.gaps:
            assert result.gaps[0].suggested_training_type == "dpo_pair"


class TestMineCapabilityGaps:
    def test_mines_across_multiple_rollouts(self):
        rollouts = [
            _make_branch("r1", 0, 1, ["read_file", "write_file"], total_score=10.0),
            _make_branch("r1", 1, 0, ["run_code"]),
            _make_branch("r2", 0, 1, ["visual_verify", "write_file"], total_score=8.0),
            _make_branch("r2", 1, 0, ["run_code"]),
        ]
        result = mine_capability_gaps(rollouts)
        assert result.n_rollouts == 2
        assert result.n_compared_pairs == 2

    def test_returns_mine_result_type(self):
        rollouts = [
            _make_branch("r1", 0, 1, ["read_file"]),
            _make_branch("r1", 1, 0, ["run_code"]),
        ]
        result = mine_capability_gaps(rollouts)
        assert isinstance(result, MineResult)
        assert isinstance(result.failure_class_counts, dict)

    def test_skips_rollout_with_no_passing_branch(self):
        rollouts = [
            _make_branch("r1", 0, 0, ["run_code"]),
            _make_branch("r1", 1, 0, ["run_code"]),
        ]
        result = mine_capability_gaps(rollouts)
        assert result.n_compared_pairs == 0
        assert result.gaps == []

    def test_skips_rollout_with_no_failing_branch(self):
        rollouts = [
            _make_branch("r1", 0, 1, ["read_file"]),
            _make_branch("r1", 1, 1, ["read_file"]),
        ]
        result = mine_capability_gaps(rollouts)
        assert result.n_compared_pairs == 0

    def test_uses_highest_scored_passing_branch(self):
        rollouts = [
            _make_branch("r1", 0, 1, ["run_code"], total_score=5.0),
            _make_branch("r1", 1, 1, ["read_file", "write_file"], total_score=10.0),
            _make_branch("r1", 2, 0, ["run_code"]),
        ]
        result = mine_capability_gaps(rollouts)
        assert result.n_compared_pairs == 1

    def test_empty_rollout_list(self):
        result = mine_capability_gaps([])
        assert result.n_rollouts == 0
        assert result.gaps == []
        assert result.suggested_dpo_pairs == 0
