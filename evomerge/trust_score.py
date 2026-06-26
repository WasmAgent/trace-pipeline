"""Agent Trust Score — composite trustworthiness metric for an agent run.

AgentTrustScore aggregates multiple evidence dimensions into a single
[0.0, 1.0] score with per-dimension breakdown. Based on the P2-1
reform roadmap formula.

Dimensions (all optional — missing dims contribute 1.0 neutral weight):
  task_success          — did the agent complete the task correctly?
  evidence_completeness — AEP evidence coverage for state-changing actions
  policy_compliance     — fraction of tool calls allowed (no deny decisions)
  budget_compliance     — did the run stay within declared budgets?
  verifier_agreement    — fraction of verifiers that passed
  benchmark_trust       — benchmark environment trust score (linter)
  supply_chain_integrity— run receipt present and verified?

Score formula: geometric mean of all present dimensions.
Any dimension scoring 0 collapses the overall score to 0.

Usage:
    from evomerge.trust_score import AgentTrustScoreBuilder, compute_trust_score

    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(aep_record_dict)
    builder.add_benchmark_trust(benchmark_trust_score)
    builder.add_receipt(has_receipt, digest_verified)
    score = builder.build()
    print(score.overall)   # 0.0 – 1.0
    print(score.breakdown) # per-dimension dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTrustScore:
    """Composite trust score for one agent run."""
    overall: float                        # geometric mean, 0.0–1.0
    breakdown: dict[str, float]          # dimension → score
    notes: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        if self.overall >= 0.9:
            return "A"
        if self.overall >= 0.75:
            return "B"
        if self.overall >= 0.6:
            return "C"
        if self.overall >= 0.4:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "grade": self.grade,
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "notes": self.notes,
        }


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    if any(v <= 0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


class AgentTrustScoreBuilder:
    """Build an AgentTrustScore incrementally from evidence sources."""

    def __init__(self) -> None:
        self._dims: dict[str, float] = {}
        self._notes: list[str] = []

    def add_aep_record(self, record: dict[str, Any]) -> "AgentTrustScoreBuilder":
        """Extract evidence_completeness, policy_compliance, verifier_agreement, budget_compliance."""
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
            self._dims["evidence_completeness"] = 1.0

        # policy_compliance: 1 - (deny_decisions / total_decisions)
        if cap_decisions:
            deny_count = sum(1 for d in cap_decisions if d.get("decision") == "deny")
            self._dims["policy_compliance"] = max(0.0, 1.0 - deny_count / len(cap_decisions))
        else:
            self._dims["policy_compliance"] = 1.0

        # verifier_agreement: fraction of verifiers passed
        if verifier_results:
            passed = sum(1 for v in verifier_results if v.get("passed", False))
            self._dims["verifier_agreement"] = passed / len(verifier_results)
        else:
            self._dims["verifier_agreement"] = 1.0

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

        return self

    def add_task_success(self, passed: bool) -> "AgentTrustScoreBuilder":
        self._dims["task_success"] = 1.0 if passed else 0.0
        return self

    def add_benchmark_trust(self, score: float) -> "AgentTrustScoreBuilder":
        """Add benchmark environment trust score (from BenchmarkTrustScore.score)."""
        self._dims["benchmark_trust"] = max(0.0, min(1.0, score))
        return self

    def add_receipt(self, has_receipt: bool, digest_verified: bool = True) -> "AgentTrustScoreBuilder":
        """Supply chain integrity: receipt present and digest verifiable."""
        if has_receipt and digest_verified:
            self._dims["supply_chain_integrity"] = 1.0
        elif has_receipt:
            self._dims["supply_chain_integrity"] = 0.7
            self._notes.append("Receipt present but digest not verified")
        else:
            self._dims["supply_chain_integrity"] = 0.5
            self._notes.append("No run receipt — supply chain integrity unverified")
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
        values = list(self._dims.values())
        overall = _geometric_mean(values)
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
