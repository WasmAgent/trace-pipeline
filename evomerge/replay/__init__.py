"""evomerge.replay — Reproducible trace replay framework (Milestone 5).

Deterministic re-execution of recorded AEP traces with side-effect mocking
and state snapshot/restore, enabling "debug mode" for failed trust checks and
regression testing of agent updates.

Quick start::

    from evomerge.replay import ReplayEngine, regression_test

    # Deterministic re-execution + reproducibility score
    result = ReplayEngine().replay(aep_record)
    print(result.determinism_score, result.is_deterministic)

    # Debug mode: stop at a failing action and inspect the state
    session = ReplayEngine().replay_debug(aep_record, stop_at_action_id="act-3")

    # Regression test: baseline agent trace vs updated agent trace
    report = regression_test(baseline_record, candidate_record)
    print(report.has_regression, report.regression_rate)
"""
from evomerge.replay.engine import (
    ActionDivergence,
    ActionOutcome,
    DebugSession,
    MockSideEffectGateway,
    RecordedSideEffect,
    RegressionReport,
    ReplayEngine,
    ReplayMismatch,
    ReplayResult,
    SideEffectCall,
    SideEffectDivergence,
    faithful_executor,
    regression_test,
)
from evomerge.replay.state import StateSnapshot, StateStore, canonical_digest

__all__ = [
    # state snapshot/restore
    "StateStore",
    "StateSnapshot",
    "canonical_digest",
    # side-effect mocking
    "MockSideEffectGateway",
    "RecordedSideEffect",
    "SideEffectCall",
    "SideEffectDivergence",
    # re-execution engine
    "ReplayEngine",
    "ReplayResult",
    "ReplayMismatch",
    "ActionOutcome",
    "DebugSession",
    "faithful_executor",
    # regression testing
    "regression_test",
    "RegressionReport",
    "ActionDivergence",
]
