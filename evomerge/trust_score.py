"""Agent Trust Score — composite trustworthiness metric for an agent run.

AgentTrustScore aggregates multiple evidence dimensions into a single
[0.0, 1.0] score with per-dimension breakdown. Based on the P2-1
reform roadmap formula.

Dimensions (all optional — missing dims are recorded as None/unknown):
  task_success          — did the agent complete the task correctly?
  evidence_completeness — AEP evidence coverage for state-changing actions
  policy_compliance     — fraction of tool calls allowed (no deny decisions)
  budget_compliance     — did the run stay within declared budgets?
  verifier_agreement    — fraction of verifiers that passed
  benchmark_trust       — benchmark environment trust score (linter)
  supply_chain_integrity— run receipt present and digest verified?

Score formula: geometric mean of all present (non-None) dimensions.
Any dimension scoring 0 collapses the overall score to 0.

Grade thresholds require a minimum number of evidence-backed dimensions:
  A: overall >= 0.9 AND >= 6 known (non-None) dimensions
  B: overall >= 0.75 AND >= 4 known dimensions
  C: overall >= 0.6
  D: overall >= 0.4
  F: otherwise

Usage:
    from evomerge.trust_score import AgentTrustScoreBuilder, compute_trust_score

    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(aep_record_dict)
    builder.add_benchmark_trust(benchmark_trust_score)
    builder.add_receipt_path(receipt_path)   # verifies digest
    score = builder.build()
    print(score.overall)   # 0.0 – 1.0 or None if no dimensions
    print(score.breakdown) # per-dimension dict (None = unknown)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Minimum number of non-None ("known") dimensions required for each grade
_MIN_KNOWN_FOR_GRADE: dict[str, int] = {
    "A": 6,
    "B": 4,
    "C": 0,
    "D": 0,
    "F": 0,
}


@dataclass
class AgentTrustScore:
    """Composite trust score for one agent run."""
    overall: float | None                     # geometric mean, 0.0–1.0, or None if no dims
    breakdown: dict[str, float | None]       # dimension → score (None = unknown)
    notes: list[str] = field(default_factory=list)

    @property
    def _known_dim_count(self) -> int:
        """Number of dimensions with a real (non-None) score."""
        return sum(1 for v in self.breakdown.values() if v is not None)

    @property
    def grade(self) -> str:
        if self.overall is None:
            return "F"
        known = self._known_dim_count
        if self.overall >= 0.9 and known >= _MIN_KNOWN_FOR_GRADE["A"]:
            return "A"
        if self.overall >= 0.75 and known >= _MIN_KNOWN_FOR_GRADE["B"]:
            return "B"
        if self.overall >= 0.6:
            return "C"
        if self.overall >= 0.4:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4) if self.overall is not None else None,
            "grade": self.grade,
            "breakdown": {
                k: (round(v, 4) if v is not None else None)
                for k, v in self.breakdown.items()
            },
            "notes": self.notes,
        }


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of a non-empty list of positive floats.

    Raises:
        ValueError: if *values* is empty — callers must handle the no-evidence case
                    explicitly rather than silently treating it as 1.0.
    """
    if not values:
        raise ValueError(
            "_geometric_mean() called with an empty list. "
            "Callers must skip the geometric mean when no evidence dimensions are present "
            "and return None (unknown) instead of 1.0."
        )
    if any(v <= 0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def _verify_receipt_digest(receipt_path: Path) -> bool:
    """Re-compute the receipt's own canonical SHA-256 and compare to receipt_digest field.

    Returns True only if the receipt file is well-formed JSON that contains a
    ``receipt_digest`` field whose value matches a freshly computed digest of
    the canonical (sort_keys, no whitespace) serialisation of the receipt body
    (i.e. all fields *except* ``receipt_digest`` itself).

    Returns False for any of: file not found, invalid JSON, missing field,
    or digest mismatch.
    """
    try:
        raw = receipt_path.read_text(encoding="utf-8")
        data: dict = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False

    stored_digest = data.get("receipt_digest")
    if not stored_digest or not isinstance(stored_digest, str):
        # Receipt does not carry a self-digest — cannot verify integrity.
        # Treat as unverified rather than passing.
        return False

    # Recompute from the receipt body (excluding the digest field itself)
    body = {k: v for k, v in data.items() if k != "receipt_digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return computed == stored_digest


class AgentTrustScoreBuilder:
    """Build an AgentTrustScore incrementally from evidence sources."""

    def __init__(self) -> None:
        self._dims: dict[str, float | None] = {}
        self._notes: list[str] = []

    def add_aep_record(self, record: dict[str, Any]) -> "AgentTrustScoreBuilder":
        """Extract evidence_completeness, policy_compliance, verifier_agreement, budget_compliance.

        When a dimension's evidence is absent (empty list / missing field), the
        dimension is recorded as *None* (unknown) rather than 1.0. This
        prevents empty records from artificially inflating the overall score.
        """
        actions = record.get("actions", [])
        cap_decisions = record.get("capability_decisions", [])
        verifier_results = record.get("verifier_results", [])
        budget = record.get("budget_ledger")

        # evidence_completeness: state-changing actions with result_digest / evidence_refs
        sc_actions = [a for a in actions if a.get("state_changing")]
        if sc_actions:
            evidenced = sum(
                1 for a in sc_actions
                if a.get("result_digest") or a.get("evidence_refs")
            )
            self._dims["evidence_completeness"] = evidenced / len(sc_actions)
        else:
            # No state-changing actions → we cannot assess evidence coverage.
            # Record as None (unknown) rather than a falsely perfect 1.0.
            self._dims["evidence_completeness"] = None
            self._notes.append(
                "evidence_completeness: no state-changing actions found — dimension unknown"
            )

        # policy_compliance: 1 - (deny_decisions / total_decisions)
        if cap_decisions:
            deny_count = sum(1 for d in cap_decisions if d.get("decision") == "deny")
            self._dims["policy_compliance"] = max(0.0, 1.0 - deny_count / len(cap_decisions))
        else:
            # No capability decisions recorded → cannot confirm policy was enforced.
            self._dims["policy_compliance"] = None
            self._notes.append(
                "policy_compliance: no capability_decisions recorded — dimension unknown"
            )

        # verifier_agreement: fraction of verifiers passed
        if verifier_results:
            passed = sum(1 for v in verifier_results if v.get("passed", False))
            self._dims["verifier_agreement"] = passed / len(verifier_results)
        else:
            # No verifier results → cannot assert correctness.
            self._dims["verifier_agreement"] = None
            self._notes.append(
                "verifier_agreement: no verifier_results recorded — dimension unknown"
            )

        # budget_compliance: check if any budget was exceeded
        if budget:
            violations = 0
            checks = 0
            tb = budget.get("token_budget")
            if tb and tb.get("limit") is not None:
                checks += 1
                if tb["spent"] > tb["limit"]:
                    violations += 1
            rb = budget.get("retry_budget")
            if rb and rb.get("limit") is not None:
                checks += 1
                if rb["spent"] > rb["limit"]:
                    violations += 1
            toolb = budget.get("tool_budget")
            if toolb and toolb.get("limit") is not None:
                checks += 1
                if toolb["spent"] > toolb["limit"]:
                    violations += 1
            if checks > 0:
                self._dims["budget_compliance"] = max(0.0, 1.0 - violations / checks)
            # If budget_ledger present but no tracked sub-budgets, leave unset (no entry).

        return self

    def add_task_success(self, passed: bool) -> "AgentTrustScoreBuilder":
        self._dims["task_success"] = 1.0 if passed else 0.0
        return self

    def add_benchmark_trust(self, score: float) -> "AgentTrustScoreBuilder":
        """Add benchmark environment trust score (from BenchmarkTrustScore.score)."""
        self._dims["benchmark_trust"] = max(0.0, min(1.0, score))
        return self

    def add_receipt(self, has_receipt: bool, digest_verified: bool = False) -> "AgentTrustScoreBuilder":
        """Supply chain integrity: receipt present and digest verified.

        DEPRECATED shorthand — prefer add_receipt_path() which performs real
        digest verification.  When called directly:
          - has_receipt=False  → 0.5  (no receipt at all)
          - has_receipt=True, digest_verified=False → 0.7 (file present, unverified)
          - has_receipt=True, digest_verified=True  → 1.0 (caller asserts verified)
        """
        if has_receipt and digest_verified:
            self._dims["supply_chain_integrity"] = 1.0
        elif has_receipt:
            self._dims["supply_chain_integrity"] = 0.7
            self._notes.append("Receipt present but digest not verified")
        else:
            self._dims["supply_chain_integrity"] = 0.5
            self._notes.append("No run receipt — supply chain integrity unverified")
        return self

    def add_receipt_path(self, receipt_path: Path | str) -> "AgentTrustScoreBuilder":
        """Add supply chain integrity evidence by loading and verifying a receipt file.

        The receipt must be a JSON object produced by RunReceiptBuilder.  This
        method recomputes the SHA-256 digest of the canonical receipt body and
        compares it against the stored ``receipt_digest`` field.

          - File not found or invalid JSON → 0.5 (no receipt)
          - File present but digest missing/mismatch → 0.7 (tampered or legacy)
          - Digest matches → 1.0 (verified)
        """
        p = Path(receipt_path)
        if not p.exists():
            self._dims["supply_chain_integrity"] = 0.5
            self._notes.append(f"Receipt file not found: {p.name} — supply chain unverified")
            return self

        if _verify_receipt_digest(p):
            self._dims["supply_chain_integrity"] = 1.0
        else:
            self._dims["supply_chain_integrity"] = 0.7
            self._notes.append(
                f"Receipt present ({p.name}) but digest verification failed — "
                "file may be tampered or lacks receipt_digest field"
            )
        return self

    def add_replay_determinism(self, score: float) -> "AgentTrustScoreBuilder":
        """Score for how deterministic/replayable the run is (0.0–1.0).
        1.0 = fully deterministic (seed set, no randomness), 0.0 = non-deterministic.
        """
        self._dims["replay_determinism"] = max(0.0, min(1.0, score))
        return self

    def add_contamination_resistance(self, score: float) -> "AgentTrustScoreBuilder":
        """Score for how resistant the benchmark/task is to contamination.
        Derived from BenchmarkTrustScore after running benchmark_linter.
        1.0 = no contamination risk detected, 0.0 = critical risks.
        """
        self._dims["contamination_resistance"] = max(0.0, min(1.0, score))
        return self

    def add_tool_misuse_resistance(self, score: float) -> "AgentTrustScoreBuilder":
        """Score for how well the runtime prevented tool misuse.
        1.0 = all risky tools were gated, 0.0 = unrestricted tool access.
        Can be derived from MCPGateway deny_rate or AEP capability_decisions.
        """
        self._dims["tool_misuse_resistance"] = max(0.0, min(1.0, score))
        return self

    def add_redaction_quality(self, score: float) -> "AgentTrustScoreBuilder":
        """Score for PII/secret redaction quality (0.0–1.0).
        1.0 = redaction profile applied and verified, 0.0 = no redaction.
        """
        self._dims["redaction_quality"] = max(0.0, min(1.0, score))
        return self

    def add_runtime_isolation(self, score: float) -> "AgentTrustScoreBuilder":
        """Score for runtime isolation level.
        1.0 = WASM sandbox with capability manifest, 0.5 = process isolation, 0.0 = no isolation.
        """
        self._dims["runtime_isolation_level"] = max(0.0, min(1.0, score))
        return self

    def add_dimension(self, name: str, score: float) -> "AgentTrustScoreBuilder":
        """Add an arbitrary named dimension (0.0–1.0)."""
        self._dims[name] = max(0.0, min(1.0, score))
        return self

    def build(self) -> AgentTrustScore:
        # Only include non-None dimensions in the geometric mean
        known_values = [v for v in self._dims.values() if v is not None]
        if known_values:
            overall: float | None = _geometric_mean(known_values)
        else:
            overall = None
            self._notes.append(
                "No evidence dimensions present — overall score is unknown (None)"
            )
        return AgentTrustScore(
            overall=overall,
            breakdown=dict(self._dims),
            notes=list(self._notes),
        )


def compute_trust_score(
    aep_record: dict[str, Any] | None = None,
    task_passed: bool | None = None,
    benchmark_trust: float | None = None,
    has_receipt: bool = False,
    replay_determinism: float | None = None,
    contamination_resistance: float | None = None,
    tool_misuse_resistance: float | None = None,
    redaction_quality: float | None = None,
    runtime_isolation_level: float | None = None,
) -> AgentTrustScore:
    """Convenience function: build trust score from common inputs."""
    builder = AgentTrustScoreBuilder()
    if aep_record is not None:
        builder.add_aep_record(aep_record)
    if task_passed is not None:
        builder.add_task_success(task_passed)
    if benchmark_trust is not None:
        builder.add_benchmark_trust(benchmark_trust)
    builder.add_receipt(has_receipt)
    if replay_determinism is not None:
        builder.add_replay_determinism(replay_determinism)
    if contamination_resistance is not None:
        builder.add_contamination_resistance(contamination_resistance)
    if tool_misuse_resistance is not None:
        builder.add_tool_misuse_resistance(tool_misuse_resistance)
    if redaction_quality is not None:
        builder.add_redaction_quality(redaction_quality)
    if runtime_isolation_level is not None:
        builder.add_runtime_isolation(runtime_isolation_level)
    return builder.build()


# Required AEP top-level fields used for schema validity check
_AEP_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "model_id",
    "repo_commit",
    "tool_manifest_digest",
    "actions",
    "capability_decisions",
    "verifier_results",
    "created_at_ms",
)


def compute_calibrated_trust_score(aep_record: dict[str, Any]) -> dict[str, Any]:
    """Compute calibrated trust score with three focused dimensions.

    Assesses an AEP record across three complementary axes and returns a
    composite score suitable for downstream training-data filtering.

    Dimensions:
      evidence_health      — Is the AEP record complete and well-formed?
                             Checks required field presence, non-empty actions
                             list, and evidence coverage on state-changing
                             actions (result_digest or evidence_refs).

      policy_risk          — Are there policy violations or suspicious
                             decisions? Penalised by deny decisions in
                             capability_decisions and by state-changing
                             actions that lack any evidence (missing digests
                             treated as an integrity risk). Score is
                             0 = high risk, 1 = no risk.

      training_eligibility — Is this record suitable for training data?
                             Combines verifier pass-rate, evidence completeness
                             on state-changing actions, and provenance signals
                             (repo_commit present, tool_manifest_digest
                             present).

    Composite formula (weighted average):
        composite = evidence_health * 0.35
                  + policy_risk     * 0.35
                  + training_eligibility * 0.30

    Labels:
        composite >= 0.8  → "high_trust"
        composite >= 0.6  → "medium_trust"
        composite >= 0.4  → "low_trust"
        composite <  0.4  → "untrusted"

    NOTE: This score is for evidence health and training eligibility
    assessment. It is NOT a compliance certification and does NOT claim
    to satisfy EU AI Act / ISO 42001 / NIST requirements.

    Args:
        aep_record: Raw AEP record dict (as produced by AEPEmitter).

    Returns:
        Dict with keys: evidence_health, policy_risk,
        training_eligibility, composite, label.
    """
    actions: list[dict] = aep_record.get("actions") or []
    cap_decisions: list[dict] = aep_record.get("capability_decisions") or []
    verifier_results: list[dict] = aep_record.get("verifier_results") or []

    # ------------------------------------------------------------------
    # 1. evidence_health
    # ------------------------------------------------------------------
    # Schema validity: all required fields present
    missing_fields = [f for f in _AEP_REQUIRED_FIELDS if f not in aep_record]
    schema_valid = 1.0 if not missing_fields else max(
        0.0, 1.0 - len(missing_fields) / len(_AEP_REQUIRED_FIELDS)
    )

    # Completeness: record has at least one action
    has_actions = 1.0 if actions else 0.0

    # Evidence coverage: state-changing actions with result_digest or evidence_refs
    sc_actions = [a for a in actions if a.get("state_changing")]
    if sc_actions:
        evidenced_count = sum(
            1 for a in sc_actions
            if a.get("result_digest") or a.get("evidence_refs")
        )
        evidence_coverage = evidenced_count / len(sc_actions)
    else:
        evidence_coverage = 1.0  # no state-changing actions → no gap

    evidence_health = (schema_valid * 0.4 + has_actions * 0.3 + evidence_coverage * 0.3)

    # ------------------------------------------------------------------
    # 2. policy_risk  (0 = high risk, 1 = no risk)
    # ------------------------------------------------------------------
    # Penalise deny decisions
    if cap_decisions:
        deny_count = sum(1 for d in cap_decisions if d.get("decision") == "deny")
        deny_penalty = deny_count / len(cap_decisions)
    else:
        deny_penalty = 0.0

    # Penalise state-changing actions without evidence (integrity risk)
    if sc_actions:
        unevidenced_count = len(sc_actions) - (
            sum(1 for a in sc_actions if a.get("result_digest") or a.get("evidence_refs"))
        )
        digest_penalty = unevidenced_count / len(sc_actions)
    else:
        digest_penalty = 0.0

    # Combine penalties with equal weight; clamp to [0, 1]
    policy_risk = max(0.0, 1.0 - (deny_penalty * 0.6 + digest_penalty * 0.4))

    # ------------------------------------------------------------------
    # 3. training_eligibility
    # ------------------------------------------------------------------
    # Verifier pass-rate
    if verifier_results:
        passed_count = sum(1 for v in verifier_results if v.get("passed", False))
        verifier_pass_rate = passed_count / len(verifier_results)
    else:
        verifier_pass_rate = 0.5  # unknown — neutral penalty

    # Evidence completeness (reuse from above)
    eligibility_evidence = evidence_coverage

    # Provenance signals: repo_commit and tool_manifest_digest present and non-empty
    provenance_score = (
        (1.0 if aep_record.get("repo_commit") else 0.0) * 0.5
        + (1.0 if aep_record.get("tool_manifest_digest") else 0.0) * 0.5
    )

    training_eligibility = (
        verifier_pass_rate * 0.45
        + eligibility_evidence * 0.30
        + provenance_score * 0.25
    )

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------
    composite = (
        evidence_health * 0.35
        + policy_risk * 0.35
        + training_eligibility * 0.30
    )
    composite = max(0.0, min(1.0, composite))

    if composite >= 0.8:
        label = "high_trust"
    elif composite >= 0.6:
        label = "medium_trust"
    elif composite >= 0.4:
        label = "low_trust"
    else:
        label = "untrusted"

    return {
        "evidence_health": round(evidence_health, 4),
        "policy_risk": round(policy_risk, 4),
        "training_eligibility": round(training_eligibility, 4),
        "composite": round(composite, 4),
        "label": label,
    }
