"""Deterministic re-execution engine for recorded AEP traces.

This is the reproducible trace replay framework (Milestone 5). Given a recorded
AEP record (its ``actions`` sequence, with ``result_digest`` /
``pre_state_digest`` / ``post_state_digest`` evidence), the engine re-executes
the trace **deterministically** in a simulated environment:

* **Side-effect mocking** — state-changing tool calls never reach a real system.
  :class:`MockSideEffectGateway` serves the recorded result for the next call
  in a cassette; if the replay requests a call the recording did not capture
  (a different tool, different args, or a re-ordering), it raises
  :class:`SideEffectDivergence`. That divergence is the signal that the replayed
  run took a different path than the recording — i.e. the trace is *not*
  reproducible.
* **State snapshot/restore** — the only thing a replayed action can mutate is a
  :class:`~evomerge.replay.state.StateStore`, which is checkpointed before each
  action. Rolling back to a checkpoint is what powers debug mode.
* **Determinism score** — fraction of actions whose replayed result/state match
  the recording (causal consistency + side-effect coverage + result fidelity).
  A cleanly recorded trace replays to 1.0; a tampered or non-reproducible trace
  scores below 1.0. Feed this into
  :meth:`~evomerge.trust_score.AgentTrustScoreBuilder.add_replay_result`.
* **Debug mode** — :meth:`ReplayEngine.replay_debug` replays up to a given
  action (or the first mismatch) and returns the live :class:`StateStore` at
  that point, so a failed trust check can be inspected step by step.
* **Regression testing** — :func:`regression_test` compares a baseline agent's
  recorded trace against an updated agent's trace and flags actions whose
  outcome changed.

The default re-executor (:func:`faithful_executor`) reproduces recorded
results; callers can inject their own ``executor`` (e.g. the updated agent's
re-implementation) to detect fidelity drift against the recording.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evomerge.replay.state import StateSnapshot, StateStore, canonical_digest

# ---------------------------------------------------------------------------
# Recorded side effects + mocking gateway
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideEffectCall:
    """Hashable identity of one recorded tool call.

    ``args_key`` is the content digest of the call's arguments (or "" when the
    action recorded none), so calls are compared order- and value-independently.
    """

    tool_name: str
    args_key: str
    action_id: str | None

    @classmethod
    def from_action(cls, action: dict[str, Any]) -> SideEffectCall:
        args = action.get("args")
        args_key = canonical_digest(args) if args else ""
        return cls(
            tool_name=action.get("tool_name", ""),
            args_key=args_key,
            action_id=action.get("action_id"),
        )


@dataclass(frozen=True)
class RecordedSideEffect:
    """One entry in a side-effect cassette: the call plus what the recording
    captured for it."""

    call: SideEffectCall
    result_digest: str | None
    post_state_digest: str | None


class SideEffectDivergence(Exception):
    """Replay requested a side effect the recording did not capture (or captured
    in a different order). The recorded trace is not reproducible against the
    current re-executor."""


class MockSideEffectGateway:
    """Deterministic stand-in for a real tool/side-effect backend during replay.

    Built from a recording's state-changing actions (:meth:`from_actions`),
    the gateway replays their captured results in order. It is the single
    chokepoint that keeps replay hermetic — a replayed run can only observe
    what the recording captured, never a live system.
    """

    def __init__(self, cassette: list[RecordedSideEffect]) -> None:
        self._cassette: list[RecordedSideEffect] = list(cassette)
        self._cursor: int = 0
        # Audit trail of every consume() request, in replay order.
        self.calls_made: list[SideEffectCall] = []

    @classmethod
    def from_actions(cls, actions: list[dict[str, Any]]) -> MockSideEffectGateway:
        cassette = [
            RecordedSideEffect(
                call=SideEffectCall.from_action(a),
                result_digest=a.get("result_digest"),
                post_state_digest=a.get("post_state_digest"),
            )
            for a in actions
            if a.get("state_changing")
        ]
        return cls(cassette)

    @property
    def cassette_len(self) -> int:
        return len(self._cassette)

    @property
    def remaining(self) -> int:
        return len(self._cassette) - self._cursor

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._cassette)

    def consume(self, action: dict[str, Any]) -> RecordedSideEffect:
        """Return the recorded side effect for ``action``'s tool call.

        The next cassette entry must match the requested call (same tool, same
        args, same action_id); otherwise the replayed run diverged from the
        recording and :class:`SideEffectDivergence` is raised. Advancing the
        cursor on a match is what makes the replay deterministic.
        """
        call = SideEffectCall.from_action(action)
        self.calls_made.append(call)
        if self._cursor >= len(self._cassette):
            raise SideEffectDivergence(
                f"replay requested an unrecorded side effect for action "
                f"{call.action_id!r} (tool={call.tool_name!r}) — cassette exhausted"
            )
        expected = self._cassette[self._cursor]
        if expected.call != call:
            raise SideEffectDivergence(
                f"side-effect divergence at cassette step {self._cursor}: "
                f"recording expected tool={expected.call.tool_name!r} "
                f"(action={expected.call.action_id!r}), but replay requested "
                f"tool={call.tool_name!r} (action={call.action_id!r})"
            )
        self._cursor += 1
        return expected


# ---------------------------------------------------------------------------
# Re-executor contract
# ---------------------------------------------------------------------------


@dataclass
class ActionOutcome:
    """Result of re-executing a single action.

    ``ok=False`` (with ``error``) means the re-execution could not faithfully
    reproduce the action — typically a :class:`SideEffectDivergence` surfacing
    an unrecorded/reordered side effect. ``ok=True`` with ``result_digest``
    is the replayed digest the engine compares against the recording.
    """

    result_digest: str | None
    ok: bool
    error: str | None = None


ActionExecutor = Callable[[dict[str, Any], StateStore, MockSideEffectGateway], ActionOutcome]


def faithful_executor(
    action: dict[str, Any],
    state: StateStore,
    gateway: MockSideEffectGateway,
) -> ActionOutcome:
    """Default re-executor: reproduces recorded results.

    State-changing actions resolve their external side effect through the mock
    gateway (so no real system is touched) and replay the recorded
    ``result_digest``; pure (non-state-changing) actions re-derive from the
    recording. A cleanly recorded trace therefore replays to a determinism
    score of 1.0 against this executor.
    """
    if action.get("state_changing"):
        try:
            effect = gateway.consume(action)
        except SideEffectDivergence as exc:
            return ActionOutcome(result_digest=None, ok=False, error=str(exc))
        if effect.post_state_digest is not None:
            state.set(
                f"post_state:{action.get('action_id')}",
                effect.post_state_digest,
            )
        return ActionOutcome(result_digest=effect.result_digest, ok=True)
    # Non-state-changing action: a pure computation — re-derive from recording.
    return ActionOutcome(result_digest=action.get("result_digest"), ok=True)


# ---------------------------------------------------------------------------
# Replay result types
# ---------------------------------------------------------------------------


@dataclass
class ReplayMismatch:
    """One action whose replay did not match the recording."""

    action_id: str | None
    kind: str  # "causal" | "fidelity" | "side_effect"
    recorded: str | None
    replayed: str | None
    detail: str


@dataclass
class ReplayResult:
    """Outcome of deterministically re-executing one AEP record."""

    run_id: str
    trace_id: str | None
    determinism_score: float
    actions_replayed: int
    actions_matched: int
    actions_total: int = 0
    mismatches: list[ReplayMismatch] = field(default_factory=list)
    snapshots: list[StateSnapshot] = field(default_factory=list)
    final_state_digest: str | None = None
    side_effect_calls: int = 0
    side_effect_divergences: int = 0
    # Debug-mode metadata (populated by replay_debug; "completed" otherwise).
    stop_reason: str = "completed"
    stop_action_id: str | None = None

    @property
    def is_deterministic(self) -> bool:
        """True iff every replayed action matched the recording."""
        return self.actions_replayed == self.actions_matched and not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "determinism_score": self.determinism_score,
            "is_deterministic": self.is_deterministic,
            "actions_replayed": self.actions_replayed,
            "actions_matched": self.actions_matched,
            "actions_total": self.actions_total,
            "final_state_digest": self.final_state_digest,
            "side_effect_calls": self.side_effect_calls,
            "side_effect_divergences": self.side_effect_divergences,
            "mismatches": [
                {
                    "action_id": m.action_id,
                    "kind": m.kind,
                    "recorded": m.recorded,
                    "replayed": m.replayed,
                    "detail": m.detail,
                }
                for m in self.mismatches
            ],
        }


# ---------------------------------------------------------------------------
# Debug session + regression report
# ---------------------------------------------------------------------------


@dataclass
class DebugSession:
    """State captured at the stop point of a debug replay."""

    result: ReplayResult
    state: StateStore
    stop_reason: str  # "action_id" | "mismatch" | "completed"
    stop_action_id: str | None


@dataclass
class ActionDivergence:
    """An action whose outcome differs between baseline and candidate."""

    action_id: str | None
    baseline_digest: str | None
    candidate_digest: str | None


@dataclass
class RegressionReport:
    """Result of comparing a baseline agent trace to an updated agent trace."""

    baseline_run_id: str
    candidate_run_id: str
    actions_compared: int
    divergences: list[ActionDivergence] = field(default_factory=list)
    baseline_determinism: float | None = None
    candidate_determinism: float | None = None

    @property
    def regression_rate(self) -> float:
        if self.actions_compared == 0:
            return 0.0
        return len(self.divergences) / self.actions_compared

    @property
    def has_regression(self) -> bool:
        return bool(self.divergences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "actions_compared": self.actions_compared,
            "regression_rate": self.regression_rate,
            "has_regression": self.has_regression,
            "baseline_determinism": self.baseline_determinism,
            "candidate_determinism": self.candidate_determinism,
            "divergences": [
                {
                    "action_id": d.action_id,
                    "baseline_digest": d.baseline_digest,
                    "candidate_digest": d.candidate_digest,
                }
                for d in self.divergences
            ],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ReplayEngine:
    """Deterministically re-executes recorded AEP traces.

    Parameters
    ----------
    seed:
        Pinned into the simulated state store so two replays of the same record
        start from byte-identical state (a prerequisite for determinism).
    initial_state:
        Optional pre-seeded state (e.g. a restored snapshot from a prior run).
    """

    def __init__(
        self,
        seed: int = 0,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self._seed = seed
        self._initial_state = dict(initial_state) if initial_state else {}

    def replay(
        self,
        record: dict[str, Any],
        executor: ActionExecutor | None = None,
        gateway: MockSideEffectGateway | None = None,
    ) -> ReplayResult:
        """Re-execute ``record`` deterministically and score reproducibility."""
        return self._run(
            record=record,
            executor=executor or faithful_executor,
            gateway=gateway,
            stop_at_action_id=None,
            stop_at_first_mismatch=False,
            debug_state=None,
        )

    def replay_debug(
        self,
        record: dict[str, Any],
        stop_at_action_id: str | None = None,
        stop_at_first_mismatch: bool = True,
        executor: ActionExecutor | None = None,
        gateway: MockSideEffectGateway | None = None,
    ) -> DebugSession:
        """Replay for "debug mode", stopping to expose intermediate state.

        Stops at ``stop_at_action_id`` (after that action's pre-snapshot, i.e.
        with the state the agent saw going into it), or at the first mismatch,
        or when the trace completes — whichever comes first. The returned
        :class:`DebugSession` carries the live :class:`StateStore` at the stop
        point, which can be rolled back to any earlier snapshot for inspection.
        """
        debug_state = StateStore(initial=self._initial_state)
        result = self._run(
            record=record,
            executor=executor or faithful_executor,
            gateway=gateway,
            stop_at_action_id=stop_at_action_id,
            stop_at_first_mismatch=stop_at_first_mismatch,
            debug_state=debug_state,
        )
        stop_id = result.mismatches[0].action_id if (
            result.stop_reason == "mismatch" and result.mismatches
        ) else stop_at_action_id
        return DebugSession(
            result=result,
            state=debug_state,
            stop_reason=result.stop_reason,
            stop_action_id=stop_id,
        )

    def _run(
        self,
        record: dict[str, Any],
        executor: ActionExecutor,
        gateway: MockSideEffectGateway | None,
        stop_at_action_id: str | None,
        stop_at_first_mismatch: bool,
        debug_state: StateStore | None,
    ) -> ReplayResult:
        actions = list(record.get("actions", []))
        gw = gateway if gateway is not None else MockSideEffectGateway.from_actions(actions)
        state = debug_state if debug_state is not None else StateStore(initial=self._initial_state)
        if "_seed" not in state.as_dict():
            state.set("_seed", self._seed)

        snapshots: list[StateSnapshot] = []
        mismatches: list[ReplayMismatch] = []
        mismatched_ids: set[str | None] = set()
        stop_reason = "completed"
        stop_action_id: str | None = None

        for action in actions:
            pre_snapshot = state.snapshot()
            snapshots.append(pre_snapshot)
            action_id = action.get("action_id")

            # Stop *before* re-executing the requested debug action so the
            # store reflects the state the agent saw going into it.
            if stop_at_action_id is not None and action_id == stop_at_action_id:
                stop_reason = "action_id"
                stop_action_id = action_id
                break

            # 1) Causal consistency: the recording's pre_state_digest (if any)
            #    must equal the current replayed state digest.
            recorded_pre = action.get("pre_state_digest")
            if recorded_pre is not None and recorded_pre != state.digest():
                mismatches.append(ReplayMismatch(
                    action_id=action_id,
                    kind="causal",
                    recorded=recorded_pre,
                    replayed=state.digest(),
                    detail=(
                        "recorded pre_state_digest does not match replayed "
                        "state — the trace is not causally reproducible"
                    ),
                ))
                mismatched_ids.add(action_id)

            # 2) Re-execute the action through the (mocked) side-effect gateway.
            outcome = executor(action, state, gw)
            if not outcome.ok:
                mismatches.append(ReplayMismatch(
                    action_id=action_id,
                    kind="side_effect",
                    recorded=action.get("result_digest"),
                    replayed=outcome.result_digest,
                    detail=outcome.error or "side-effect resolution failed",
                ))
                mismatched_ids.add(action_id)
            else:
                # 3) Result fidelity: replayed digest must equal the recording.
                recorded_result = action.get("result_digest")
                replayed_result = outcome.result_digest
                if (
                    recorded_result is not None
                    and replayed_result is not None
                    and recorded_result != replayed_result
                ):
                    mismatches.append(ReplayMismatch(
                        action_id=action_id,
                        kind="fidelity",
                        recorded=recorded_result,
                        replayed=replayed_result,
                        detail="replayed result_digest differs from recording",
                    ))
                    mismatched_ids.add(action_id)

            if stop_at_first_mismatch and action_id in mismatched_ids:
                # Rewind to the pre-snapshot so debug state shows what the
                # agent saw at the failing step.
                state.restore(pre_snapshot)
                stop_reason = "mismatch"
                stop_action_id = action_id
                break

        total = len(actions)
        # If we broke before executing the debug stop-target, drop its
        # pre-snapshot so neither "replayed" nor "matched" count it.
        if stop_reason == "action_id":
            snapshots.pop()
        # "replayed" counts every action we took a pre-snapshot for and began
        # processing; "matched" excludes any that produced a mismatch.
        replayed_count = len(snapshots)
        matched = replayed_count - len(mismatched_ids)
        score = 1.0 if replayed_count == 0 else matched / replayed_count

        result = ReplayResult(
            run_id=record.get("run_id", "<unknown>"),
            trace_id=record.get("trace_id"),
            determinism_score=score,
            actions_replayed=replayed_count,
            actions_matched=matched,
            actions_total=total,
            mismatches=mismatches,
            snapshots=snapshots,
            final_state_digest=state.digest(),
            side_effect_calls=len(gw.calls_made),
            side_effect_divergences=sum(
                1 for m in mismatches if m.kind == "side_effect"
            ),
            stop_reason=stop_reason,
            stop_action_id=stop_action_id,
        )
        return result


def regression_test(
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
    executor: ActionExecutor | None = None,
) -> RegressionReport:
    """Detect regressions between a baseline agent trace and an updated one.

    Both traces are deterministically replayed; for every action present in
    both recordings (matched by ``action_id``), the replayed ``result_digest``
    is compared. A change in outcome on the candidate side is a regression —
    the agent update altered behaviour for a previously-recorded step. Both
    sides' determinism scores are included so a regression caused by the
    candidate becoming non-reproducible is also visible.
    """
    engine = ReplayEngine()
    baseline_result = engine.replay(baseline_record, executor=executor)
    candidate_result = engine.replay(candidate_record, executor=executor)

    def _digests_by_action(record: dict[str, Any]) -> dict[str | None, str | None]:
        return {
            a.get("action_id"): a.get("result_digest")
            for a in record.get("actions", [])
        }

    base_digests = _digests_by_action(baseline_record)
    cand_digests = _digests_by_action(candidate_record)
    common = list(base_digests.keys() & cand_digests.keys())

    divergences = [
        ActionDivergence(
            action_id=aid,
            baseline_digest=base_digests[aid],
            candidate_digest=cand_digests[aid],
        )
        for aid in common
        if base_digests[aid] != cand_digests[aid]
    ]

    return RegressionReport(
        baseline_run_id=baseline_record.get("run_id", "<unknown>"),
        candidate_run_id=candidate_record.get("run_id", "<unknown>"),
        actions_compared=len(common),
        divergences=divergences,
        baseline_determinism=baseline_result.determinism_score,
        candidate_determinism=candidate_result.determinism_score,
    )
