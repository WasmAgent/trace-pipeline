"""evomerge.streaming — incremental (streaming) compliance evaluation.

This module is the **streaming evaluation layer** of the Milestone-5 production
trace pipeline (issue #52). It replaces batch-mode constraint checking —
``validate_aep_file`` followed by ``admission_gate`` over an entire record list —
with incremental evaluation as AEP records arrive one at a time.

Design
  • ``StreamingComplianceEvaluator.ingest(record)`` evaluates a single record
    against an ordered constraint pipeline and returns a ``StreamingFeedback``
    immediately. Hard constraints are checked first and **short-circuit**: the
    first hard violation terminates the remaining (more expensive) checks so
    feedback reaches the agent control loop in well under a second
    ("early detection of violations").
  • Each ingest is timed; ``StreamingStats`` aggregates latency so callers can
    monitor the sub-second SLA — ``within_sla`` fraction and ``p99_latency_ms``.
  • Stateful cross-record constraints detect patterns that only emerge across a
    stream — e.g. a subject accumulating hard violations past a budget emits a
    backpressure advisory so the control loop can throttle or quarantine it.
  • Stateless per-record checks reuse the existing ``evomerge.validate``
    primitives (``validate_aep_record``, ``check_anomalous_scores``,
    ``check_injection_signals``) so streaming and batch evaluation agree on what
    counts as a violation — see ``tests/test_streaming.py::TestBatchParity``.

The evaluator is deliberately dependency-free (no asyncio, no queue library):
the "stream" is a plain iterator of record dicts, which the CLI consumes line by
line. Backpressure-aware queueing (Redis/RabbitMQ) and concurrent validation
workers belong to the ingestion-scaling bullet (Milestone 5, issue #51) and live
outside this repo — this layer consumes whatever iterator the ingestion tier
hands it.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from evomerge.validate.aep import validate_aep_record
from evomerge.validate.quality_gate import (
    INJECTION_SIGNAL_FRAGMENTS,
    check_anomalous_scores,
    check_injection_signals,
)

#: Per-ingest latency SLA in milliseconds. The milestone requires sub-second
#: feedback to agent control loops; an ingest above this breaches the SLA and is
#: surfaced via ``StreamingStats.within_sla``.
SLA_LATENCY_MS: float = 1000.0

VERDICT_PASS = "pass"
VERDICT_QUARANTINE = "quarantine"
VERDICT_REJECT = "reject"


# ===========================================================================
# Result types
# ===========================================================================


@dataclass
class StreamingViolation:
    """A single constraint violation detected for one streaming record.

    Shape mirrors :class:`evomerge.schemas.compliance.ConstraintViolation` so
    streaming results drop into the existing compliance export pipeline.
    """

    constraint_id: str
    level: str            # "hard" | "soft"
    category: str
    hint: str
    action_id: str | None = None


@dataclass
class StreamingFeedback:
    """Immediate evaluation result for one ingested record.

    An agent control loop acts on ``verdict`` the instant this returns — it does
    not wait for the rest of the stream to be consumed.
    """

    run_id: str
    sequence: int
    verdict: str          # VERDICT_PASS | VERDICT_QUARANTINE | VERDICT_REJECT
    violations: list[StreamingViolation] = field(default_factory=list)
    latency_ms: float = 0.0
    early_terminated: bool = False
    score: float | None = None
    admission_category: str | None = None

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_PASS

    @property
    def hard_violations(self) -> list[StreamingViolation]:
        return [v for v in self.violations if v.level == "hard"]

    @property
    def soft_violations(self) -> list[StreamingViolation]:
        return [v for v in self.violations if v.level == "soft"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "verdict": self.verdict,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 3),
            "early_terminated": self.early_terminated,
            "score": self.score,
            "admission_category": self.admission_category,
            "violations": [
                {"constraint_id": v.constraint_id, "level": v.level,
                 "category": v.category, "hint": v.hint, "action_id": v.action_id}
                for v in self.violations
            ],
        }


@dataclass
class StreamingStats:
    """Cumulative counters and latency aggregation across the stream."""

    ingested: int = 0
    passed: int = 0
    quarantined: int = 0
    rejected: int = 0
    hard_violations: int = 0
    soft_violations: int = 0
    early_terminations: int = 0
    _latencies_ms: list[float] = field(default_factory=list, repr=False)

    @property
    def n_violations(self) -> int:
        return self.hard_violations + self.soft_violations

    @property
    def violation_rate(self) -> float:
        """Fraction of ingested records that did not pass (quarantined + rejected)."""
        if self.ingested == 0:
            return 0.0
        return (self.rejected + self.quarantined) / self.ingested

    @property
    def mean_latency_ms(self) -> float:
        return sum(self._latencies_ms) / len(self._latencies_ms) if self._latencies_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self._latencies_ms) if self._latencies_ms else 0.0

    def latency_quantile(self, q: float) -> float:
        """Nearest-rank latency quantile in milliseconds (``q`` in [0, 1])."""
        if not self._latencies_ms:
            return 0.0
        ordered = sorted(self._latencies_ms)
        rank = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
        return ordered[rank]

    @property
    def p99_latency_ms(self) -> float:
        return self.latency_quantile(0.99)

    @property
    def within_sla(self) -> float:
        """Fraction of ingests that completed under ``SLA_LATENCY_MS``."""
        if not self._latencies_ms:
            return 1.0
        good = sum(1 for x in self._latencies_ms if x <= SLA_LATENCY_MS)
        return good / len(self._latencies_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingested": self.ingested,
            "passed": self.passed,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "hard_violations": self.hard_violations,
            "soft_violations": self.soft_violations,
            "early_terminations": self.early_terminations,
            "violation_rate": round(self.violation_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "sla_latency_ms": SLA_LATENCY_MS,
            "within_sla": round(self.within_sla, 4),
        }

    def _record(self, feedback: StreamingFeedback) -> None:
        self.ingested += 1
        self._latencies_ms.append(feedback.latency_ms)
        self.hard_violations += len(feedback.hard_violations)
        self.soft_violations += len(feedback.soft_violations)
        if feedback.early_terminated:
            self.early_terminations += 1
        if feedback.verdict == VERDICT_PASS:
            self.passed += 1
        elif feedback.verdict == VERDICT_QUARANTINE:
            self.quarantined += 1
        else:
            self.rejected += 1


# ===========================================================================
# Record helpers
# ===========================================================================


def _subject_of(record: dict[str, Any]) -> str:
    """Best-effort subject identity for the stateful violation budget.

    AEP v0.3 nests ``subject_id`` under ``run_context``; v0.1/v0.2 and rollout
    records carry it (or ``user_id``) at the top level. Falls back to ``run_id``
    so every record is attributable to *some* subject.
    """
    for key in ("subject_id", "user_id"):
        if record.get(key):
            return str(record[key])
    ctx = record.get("run_context")
    if isinstance(ctx, dict):
        for key in ("subject_id", "user_id"):
            if ctx.get(key):
                return str(ctx[key])
    return str(record.get("run_id") or record.get("trace_id") or "<unknown>")


def _text_fields(record: dict[str, Any]) -> list[str]:
    """Collect free-text fields from a record for injection-signal scanning.

    AEP records are evidence-shaped (digests, refs) and rarely carry free text,
    so for a clean record this returns an empty list and the injection check is a
    no-op — never a false positive.
    """
    texts: list[str] = []
    for key in ("task", "final_answer", "prompt", "user_message"):
        val = record.get(key)
        if isinstance(val, str) and val:
            texts.append(val)
    for action in record.get("actions") or []:
        if not isinstance(action, dict):
            continue
        for key in ("input", "output", "description"):
            val = action.get(key)
            if isinstance(val, str) and val:
                texts.append(val)
    return texts


# ===========================================================================
# Evaluator
# ===========================================================================


class StreamingComplianceEvaluator:
    """Incremental compliance evaluator for streaming AEP records.

    Replaces batch-mode checking (``validate_aep_file`` then ``admission_gate``
    over the whole list) with per-record incremental evaluation. Construction is
    cheap; aside from the rolling per-subject violation budget the evaluator is
    stateless, so a long-lived instance can sit in front of an agent control loop
    and call :meth:`ingest` for each record the moment it arrives.

    Parameters
    ----------
    require_signature:
        Forwarded to :func:`validate_aep_record`; when True a missing or invalid
        Ed25519 signature is a hard violation.
    score_admission:
        When True, :func:`compute_admission_score` is attached to each feedback
        so streaming output is drop-in compatible with the batch admission gate.
        Skipped on hard-violation short-circuit (early detection takes priority).
    min_evidence_completeness:
        Floor for the soft ``evidence_complete`` check. State-changing runs whose
        evidence fraction falls below this are quarantined, not rejected.
    injection_fragments:
        Override the substring signal list (defaults to
        :data:`INJECTION_SIGNAL_FRAGMENTS`).
    max_hard_per_subject:
        Stateful backpressure budget — once a subject accumulates more than this
        many hard violations, its subsequent records carry a soft
        ``subject_violation_budget`` advisory so the control loop can throttle it.
    """

    def __init__(
        self,
        *,
        require_signature: bool = False,
        score_admission: bool = True,
        min_evidence_completeness: float = 0.8,
        injection_fragments: tuple[str, ...] | None = None,
        max_hard_per_subject: int = 3,
    ) -> None:
        self.require_signature = require_signature
        self.score_admission = score_admission
        self.min_evidence_completeness = min_evidence_completeness
        self.injection_fragments = injection_fragments or INJECTION_SIGNAL_FRAGMENTS
        self.max_hard_per_subject = max_hard_per_subject
        self.stats = StreamingStats()
        self._seq = 0
        self._subject_hard: dict[str, int] = {}

    # -- public API ---------------------------------------------------------

    def ingest(self, record: dict[str, Any]) -> StreamingFeedback:
        """Evaluate one record incrementally and return immediate feedback.

        Constraints are applied in severity order. A hard violation sets
        ``early_terminated`` and skips the remaining checks so feedback is
        returned as fast as possible (sub-second SLA, see ``StreamingStats``).
        """
        self._seq += 1
        sequence = self._seq
        run_id = str(record.get("run_id", record.get("trace_id", f"<seq-{sequence}>")))
        t0 = time.perf_counter()

        violations: list[StreamingViolation] = []
        early = False

        # 1. Schema validity (hard) — short-circuits everything else.
        aep_result = validate_aep_record(record, require_signature=self.require_signature)
        if not aep_result.valid_schema:
            schema_errs = [e for e in aep_result.errors if e.startswith("schema:")]
            violations.append(StreamingViolation(
                constraint_id="aep_schema", level="hard", category="format",
                hint=schema_errs[0] if schema_errs else "AEP schema validation failed",
            ))
            early = True
        elif self.require_signature:
            sig_errs = [e for e in aep_result.errors if e.startswith("signature:")]
            if sig_errs:
                violations.append(StreamingViolation(
                    constraint_id="aep_signature", level="hard", category="security",
                    hint=sig_errs[0],
                ))
                early = True

        # 2. Hard security checks — only when the record is structurally valid.
        if not early:
            # Anomalous objective_score: rollout-wire records carry it; AEP
            # records do not, so this is a no-op for pure AEP streams.
            anomalous = check_anomalous_scores([record])
            if anomalous:
                violations.append(StreamingViolation(
                    constraint_id="objective_score_in_range", level="hard",
                    category="security", hint=anomalous[0].message,
                ))
                early = True

        if not early:
            texts = _text_fields(record)
            if texts:
                injected = check_injection_signals(texts, fragments=self.injection_fragments)
                if injected:
                    violations.append(StreamingViolation(
                        constraint_id="no_injection_signal", level="hard",
                        category="security", hint=injected[0].message,
                    ))
                    early = True

        # 3. Soft checks — advisory, never short-circuit.
        if not early and aep_result.state_changing_actions_total > 0 \
                and aep_result.evidence_completeness < self.min_evidence_completeness:
            violations.append(StreamingViolation(
                constraint_id="evidence_complete", level="soft", category="evidence",
                hint=(
                    f"evidence completeness {aep_result.evidence_completeness:.0%} "
                    f"below floor {self.min_evidence_completeness:.0%} for "
                    f"{aep_result.state_changing_actions_total} state-changing action(s)"
                ),
            ))

        # 4. Stateful backpressure budget (incremental across the stream).
        subject = _subject_of(record)
        if violations and violations[0].level == "hard":
            self._subject_hard[subject] = self._subject_hard.get(subject, 0) + 1
        if self._subject_hard.get(subject, 0) > self.max_hard_per_subject:
            violations.append(StreamingViolation(
                constraint_id="subject_violation_budget", level="soft", category="policy",
                hint=(
                    f"subject {subject!r} exceeded hard-violation budget "
                    f"({self._subject_hard[subject]}>{self.max_hard_per_subject}) — "
                    f"throttle/quarantine advised"
                ),
            ))

        # Verdict.
        if any(v.level == "hard" for v in violations):
            verdict = VERDICT_REJECT
        elif violations:
            verdict = VERDICT_QUARANTINE
        else:
            verdict = VERDICT_PASS

        # Admission score — skipped on early termination (feedback first).
        score: float | None = None
        admission_category: str | None = None
        if self.score_admission and not early:
            try:
                from evomerge.validate.quality_gate import compute_admission_score
                scored = compute_admission_score(record)
                score = scored.get("score")
                admission_category = scored.get("category")
            except Exception:  # pragma: no cover - admission is best-effort
                pass

        latency_ms = (time.perf_counter() - t0) * 1000.0
        feedback = StreamingFeedback(
            run_id=run_id,
            sequence=sequence,
            verdict=verdict,
            violations=violations,
            latency_ms=latency_ms,
            early_terminated=early,
            score=score,
            admission_category=admission_category,
        )
        self.stats._record(feedback)
        return feedback

    def ingest_many(
        self, records: Iterable[dict[str, Any]]
    ) -> Iterator[StreamingFeedback]:
        """Stream-evaluate records one at a time, yielding feedback each time.

        This is the streaming replacement for
        ``[validate_aep_record(r) for r in records]``: records are consumed
        lazily, so a generator-backed source (a live trace tap) is never buffered
        into memory.
        """
        for record in records:
            yield self.ingest(record)

    def reset(self) -> None:
        """Clear cumulative stats and per-subject state (e.g. between runs)."""
        self.stats = StreamingStats()
        self._seq = 0
        self._subject_hard.clear()


__all__ = [
    "SLA_LATENCY_MS",
    "StreamingComplianceEvaluator",
    "StreamingFeedback",
    "StreamingStats",
    "StreamingViolation",
    "VERDICT_PASS",
    "VERDICT_QUARANTINE",
    "VERDICT_REJECT",
]
