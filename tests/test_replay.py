"""Tests for the reproducible trace replay framework (evomerge.replay).

Covers each capability required by the Milestone 5 bullet:
  - deterministic re-execution of recorded AEP traces
  - side-effect mocking (real backends never invoked during replay)
  - state snapshot/restore
  - "debug mode" for failed trust checks (stop at an action / first mismatch)
  - regression testing of agent updates
  - trust-score integration (replay_determinism derived from a ReplayResult)
"""
from __future__ import annotations

import copy
import json

import pytest

from evomerge.replay import (
    ActionOutcome,
    MockSideEffectGateway,
    ReplayEngine,
    SideEffectDivergence,
    StateStore,
    regression_test,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _act(
    aid: str,
    tool: str,
    *,
    state_changing: bool = False,
    result_digest: str | None = None,
    args: dict | None = None,
    post_state_digest: str | None = None,
    pre_state_digest: str | None = None,
) -> dict:
    a: dict = {
        "action_id": aid,
        "tool_name": tool,
        "state_changing": state_changing,
        "timestamp_ms": 0,
    }
    if result_digest is not None:
        a["result_digest"] = result_digest
    if args is not None:
        a["args"] = args
    if post_state_digest is not None:
        a["post_state_digest"] = post_state_digest
    if pre_state_digest is not None:
        a["pre_state_digest"] = pre_state_digest
    return a


def _record(actions: list[dict], run_id: str = "r1") -> dict:
    return {
        "schema_version": "aep/v0.1",
        "run_id": run_id,
        "trace_id": f"trace-{run_id}",
        "created_at_ms": 0,
        "actions": actions,
    }


CLEAN_ACTIONS = [
    _act("a0", "execute_code", result_digest="d0"),
    _act("a1", "write_file", state_changing=True,
         result_digest="d1", post_state_digest="ps1"),
    _act("a2", "execute_code", result_digest="d2"),
]


# ---------------------------------------------------------------------------
# StateStore — snapshot / restore
# ---------------------------------------------------------------------------


def test_state_store_snapshot_restore():
    store = StateStore()
    store.set("k", "v1")
    snap = store.snapshot()

    assert snap.digest == store.digest()
    store.set("k", "v2")
    assert store.get("k") == "v2"
    assert store.digest() != snap.digest

    store.restore(snap)
    assert store.get("k") == "v1"
    assert store.digest() == snap.digest
    # Sequence counter rolls back with the contents.
    assert store.seq == snap.seq


def test_state_store_digest_is_order_independent():
    # Canonical digest must not depend on insertion order — two replays that
    # build equivalent state must produce the same digest.
    a = StateStore({"x": 1, "y": 2})
    b = StateStore({"y": 2, "x": 1})
    assert a.digest() == b.digest()


def test_state_store_snapshots_history_survives_restore():
    store = StateStore()
    store.set("k", 1)
    s0 = store.snapshot()
    store.set("k", 2)
    s1 = store.snapshot()
    store.restore(s0)
    # Both checkpoints remain available for further rollbacks.
    assert store.snapshots == [s0, s1]
    store.restore(s1)
    assert store.get("k") == 2


# ---------------------------------------------------------------------------
# MockSideEffectGateway — side-effect mocking
# ---------------------------------------------------------------------------


def test_gateway_serves_recorded_results_in_order():
    actions = [
        _act("a0", "write_file", state_changing=True, result_digest="r0"),
        _act("a1", "send_email", state_changing=True, result_digest="r1"),
    ]
    gw = MockSideEffectGateway.from_actions(actions)
    assert gw.cassette_len == 2
    assert not gw.exhausted

    assert gw.consume(actions[0]).result_digest == "r0"
    assert gw.consume(actions[1]).result_digest == "r1"
    assert gw.exhausted

    # An extra, unrecorded call → divergence (cassette exhausted).
    with pytest.raises(SideEffectDivergence, match="cassette exhausted"):
        gw.consume(actions[0])


def test_gateway_detects_reordering_as_divergence():
    actions = [
        _act("a0", "tool_a", state_changing=True, result_digest="r0"),
        _act("a1", "tool_b", state_changing=True, result_digest="r1"),
    ]
    gw = MockSideEffectGateway.from_actions(actions)
    # Replaying the second call first is a different path than recorded.
    with pytest.raises(SideEffectDivergence, match="divergence"):
        gw.consume(actions[1])


def test_replay_never_invokes_real_side_effects():
    """The headline property of side-effect mocking: a real backend is never
    touched during replay — only the recorded cassette is consulted."""
    real_calls = {"n": 0}

    def real_backend(action):
        real_calls["n"] += 1
        return f"live-{real_calls['n']}"

    def executor(action, state, gateway):
        if action.get("state_changing"):
            # Mocked: resolve from the cassette, NOT real_backend.
            effect = gateway.consume(action)
            return ActionOutcome(result_digest=effect.result_digest, ok=True)
        return ActionOutcome(result_digest=action.get("result_digest"), ok=True)

    record = _record([
        _act("a0", "write_file", state_changing=True,
             result_digest="recorded-d", post_state_digest="ps"),
    ])
    result = ReplayEngine().replay(record, executor=executor)

    assert real_calls["n"] == 0  # real backend never invoked
    assert result.side_effect_calls == 1
    assert result.is_deterministic


# ---------------------------------------------------------------------------
# Deterministic re-execution
# ---------------------------------------------------------------------------


def test_clean_trace_replays_as_deterministic():
    result = ReplayEngine().replay(_record(CLEAN_ACTIONS))
    assert result.determinism_score == 1.0
    assert result.is_deterministic
    assert result.actions_replayed == 3
    assert result.actions_matched == 3
    assert result.side_effect_calls == 1  # only a1 is state-changing
    assert result.mismatches == []


def test_replay_is_deterministic_across_runs():
    """Replaying the same recorded trace twice yields identical state + score."""
    record = _record(CLEAN_ACTIONS)
    r1 = ReplayEngine(seed=42).replay(record)
    r2 = ReplayEngine(seed=42).replay(record)
    assert r1.determinism_score == r2.determinism_score == 1.0
    assert r1.final_state_digest == r2.final_state_digest


def test_empty_trace_is_vacuously_deterministic():
    result = ReplayEngine().replay(_record([]))
    assert result.determinism_score == 1.0
    assert result.actions_replayed == 0
    assert result.is_deterministic


def test_replay_snapshots_before_each_action():
    result = ReplayEngine().replay(_record(CLEAN_ACTIONS))
    # One pre-snapshot per action, available for debug-mode rollback.
    assert len(result.snapshots) == 3
    assert all(s.digest for s in result.snapshots)


# ---------------------------------------------------------------------------
# Mismatch detection: causal / side-effect / fidelity
# ---------------------------------------------------------------------------


def test_causal_break_detected():
    record = _record([
        _act("a0", "execute_code", result_digest="d0"),
        _act("a1", "execute_code", result_digest="d1", pre_state_digest="WRONG"),
    ])
    result = ReplayEngine().replay(record)
    assert result.determinism_score < 1.0
    causal = [m for m in result.mismatches if m.kind == "causal"]
    assert len(causal) == 1
    assert causal[0].action_id == "a1"
    assert not result.is_deterministic


def test_side_effect_divergence_detected():
    record = _record([
        _act("a0", "write_file", state_changing=True,
             result_digest="d0", post_state_digest="ps0"),
    ])
    # Tampered cassette built from a different tool than the recording.
    tampered = MockSideEffectGateway.from_actions(
        [_act("a0", "delete_file", state_changing=True, result_digest="d0")]
    )
    result = ReplayEngine().replay(record, gateway=tampered)
    assert result.determinism_score < 1.0
    assert result.side_effect_divergences == 1
    assert any(m.kind == "side_effect" for m in result.mismatches)


def test_fidelity_drift_detected_via_custom_executor():
    """An updated agent whose re-execution yields a different result_digest
    for a recorded step is flagged as a fidelity mismatch."""
    def drifting_executor(action, state, gateway):
        if action.get("state_changing"):
            effect = gateway.consume(action)
            return ActionOutcome(result_digest=effect.result_digest, ok=True)
        # Pure action: the "updated agent" produces a different result.
        return ActionOutcome(result_digest="DRIFTED", ok=True)

    record = _record([_act("a0", "execute_code", result_digest="d0")])
    result = ReplayEngine().replay(record, executor=drifting_executor)
    assert result.determinism_score == 0.0
    fidelity = [m for m in result.mismatches if m.kind == "fidelity"]
    assert len(fidelity) == 1
    assert fidelity[0].recorded == "d0"
    assert fidelity[0].replayed == "DRIFTED"


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


def test_debug_mode_stops_at_requested_action():
    record = _record(CLEAN_ACTIONS)
    session = ReplayEngine().replay_debug(
        record, stop_at_action_id="a1", stop_at_first_mismatch=False
    )
    assert session.stop_reason == "action_id"
    assert session.stop_action_id == "a1"
    # a0 ran (pure, no state mutation); a1 not yet executed → its post-state
    # has not been applied to the store.
    assert session.state.get("post_state:a1") is None
    assert session.state.get("_seed") == 0
    # Earlier checkpoints remain available for rollback.
    assert len(session.state.snapshots) >= 1
    # Only a0 counts as cleanly replayed before the stop.
    assert session.result.actions_replayed == 1
    assert session.result.actions_matched == 1


def test_debug_mode_stops_at_first_mismatch():
    record = _record([
        _act("a0", "execute_code", result_digest="d0"),
        _act("a1", "execute_code", result_digest="d1", pre_state_digest="WRONG"),
        _act("a2", "execute_code", result_digest="d2"),
    ])
    session = ReplayEngine().replay_debug(record, stop_at_first_mismatch=True)
    assert session.stop_reason == "mismatch"
    assert session.stop_action_id == "a1"
    # State rolled back to a1's pre-snapshot; a2 never reached.
    assert session.state.get("post_state:a1") is None
    assert session.result.actions_total == 3
    assert session.result.actions_replayed == 2  # a0 + a1(attempted)
    assert session.result.actions_matched == 1   # only a0 clean


def test_debug_session_state_can_be_rolled_back():
    record = _record(CLEAN_ACTIONS)
    session = ReplayEngine().replay_debug(
        record, stop_at_action_id="a2", stop_at_first_mismatch=False
    )
    snaps = session.state.snapshots
    # Roll the debug state back to the very first checkpoint.
    session.state.restore(snaps[0])
    assert session.state.digest() == snaps[0].digest


# ---------------------------------------------------------------------------
# Regression testing
# ---------------------------------------------------------------------------


def test_regression_test_flags_changed_outcome():
    baseline = _record([
        _act("a0", "execute_code", result_digest="d0"),
        _act("a1", "write_file", state_changing=True,
             result_digest="d1", post_state_digest="ps1"),
    ], run_id="base")
    candidate = _record([
        _act("a0", "execute_code", result_digest="d0"),
        _act("a1", "write_file", state_changing=True,
             result_digest="d1_CHANGED", post_state_digest="ps1"),
    ], run_id="cand")

    report = regression_test(baseline, candidate)
    assert report.has_regression
    assert report.actions_compared == 2
    assert len(report.divergences) == 1
    assert report.divergences[0].action_id == "a1"
    assert report.regression_rate == 0.5


def test_regression_test_no_divergence_for_identical_outcomes():
    baseline = _record(CLEAN_ACTIONS, run_id="base")
    candidate = _record(copy.deepcopy(CLEAN_ACTIONS), run_id="cand")
    report = regression_test(baseline, candidate)
    assert not report.has_regression
    assert report.regression_rate == 0.0
    assert report.baseline_determinism == 1.0
    assert report.candidate_determinism == 1.0


def test_regression_report_serializes():
    baseline = _record([_act("a0", "execute_code", result_digest="d0")], run_id="b")
    candidate = _record([_act("a0", "execute_code", result_digest="dX")], run_id="c")
    report = regression_test(baseline, candidate)
    payload = report.to_dict()
    # Round-trips through JSON.
    json.loads(json.dumps(payload))
    assert payload["has_regression"] is True


# ---------------------------------------------------------------------------
# Trust-score integration
# ---------------------------------------------------------------------------


def test_trust_score_clean_replay_result():
    from evomerge.trust_score import AgentTrustScoreBuilder

    result = ReplayEngine().replay(_record(CLEAN_ACTIONS))
    builder = AgentTrustScoreBuilder()
    builder.add_replay_result(result)
    score = builder.build()
    assert score.breakdown["replay_determinism"] == 1.0
    assert not any("non-deterministic" in n.lower() for n in score.notes)


def test_trust_score_non_deterministic_replay_result():
    from evomerge.trust_score import AgentTrustScoreBuilder

    record = _record([
        _act("a0", "execute_code", result_digest="d0"),
        _act("a1", "execute_code", result_digest="d1", pre_state_digest="WRONG"),
    ])
    result = ReplayEngine().replay(record)
    builder = AgentTrustScoreBuilder()
    builder.add_replay_result(result)
    score = builder.build()
    assert score.breakdown["replay_determinism"] < 1.0
    assert any("non-deterministic" in n.lower() for n in score.notes)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_replay_score(tmp_path):
    from evomerge.__main__ import main

    aep = tmp_path / "aep.jsonl"
    aep.write_text(json.dumps(_record(CLEAN_ACTIONS)) + "\n")
    out = tmp_path / "out.json"
    rc = main(["replay", "--aep", str(aep), "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "score"
    assert payload["runs"][0]["run_id"] == "r1"
    assert payload["runs"][0]["determinism_score"] == 1.0


def test_cli_replay_debug(tmp_path):
    from evomerge.__main__ import main

    aep = tmp_path / "aep.jsonl"
    aep.write_text(json.dumps(_record(CLEAN_ACTIONS)) + "\n")
    out = tmp_path / "out.json"
    rc = main(["replay", "--aep", str(aep),
               "--debug-action", "a1", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "debug"
    assert payload["stop_action_id"] == "a1"
    assert "state_at_stop" in payload


def test_cli_replay_missing_file(tmp_path, capsys):
    from evomerge.__main__ import main

    rc = main(["replay", "--aep", str(tmp_path / "nope.jsonl")])
    assert rc == 1
