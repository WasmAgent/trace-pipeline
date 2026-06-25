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
    return builder.build()
