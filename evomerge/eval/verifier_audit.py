"""evomerge.eval.verifier_audit — verifier strength audit module.

Audits a collection of verifier results to surface systematic weaknesses:
degenerate pass-rates, missing score signal, binary-only scores, missing
claim linkage, and oracle-leakage risk.

Typical usage:

    from evomerge.eval.verifier_audit import audit_verifier_results, audit_aep_verifiers

    results = [
        {"passed": True, "score": 0.9, "claim_ids": ["c1"]},
        {"passed": False, "score": 0.3, "claim_ids": []},
    ]
    audit = audit_verifier_results("my_verifier", results)
    print(audit.strength_score)   # float 0-1
    print(audit.to_dict())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class VerifierAuditFinding:
    """One finding from a verifier strength audit.

    Attributes:
        severity: "critical" | "high" | "medium" | "low"
        check_id: machine-readable identifier for this check.
        description: human-readable description of the issue found.
        evidence: optional supporting data string.
        recommendation: optional remediation suggestion.
    """

    severity: str
    check_id: str
    description: str
    evidence: str = ""
    recommendation: str = ""


@dataclass
class VerifierAuditResult:
    """Aggregated audit result for a single verifier.

    Attributes:
        verifier_id: identifier of the verifier being audited.
        total_results: number of result records audited.
        pass_rate: fraction of results with passed == True.
        findings: list of VerifierAuditFinding.
    """

    verifier_id: str
    total_results: int
    pass_rate: float
    findings: list[VerifierAuditFinding] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        """Number of critical-severity findings."""
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        """Number of high-severity findings."""
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def strength_score(self) -> float:
        """Float in [0, 1] representing verifier health.

        Starts at 1.0, penalized by:
          - 0.30 per critical finding
          - 0.15 per high finding
        Clamped to [0.0, 1.0].
        """
        penalty = self.critical_count * 0.30 + self.high_count * 0.15
        return max(0.0, min(1.0, 1.0 - penalty))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for JSON export."""
        return {
            "verifier_id": self.verifier_id,
            "total_results": self.total_results,
            "pass_rate": self.pass_rate,
            "strength_score": self.strength_score,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "findings": [
                {
                    "severity": f.severity,
                    "check_id": f.check_id,
                    "description": f.description,
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
        }


def audit_verifier_results(
    verifier_id: str,
    results: Sequence[dict[str, Any]],
    task_records: Sequence[dict[str, Any]] | None = None,
) -> VerifierAuditResult:
    """Audit a list of verifier result dicts for systematic weaknesses.

    Args:
        verifier_id: identifier of the verifier (used for oracle-leakage check).
        results: sequence of result dicts, each expected to contain:
            - "passed" (bool)
            - "score" (float | None)
            - "claim_ids" (list[str] | None)
        task_records: optional parallel task records (reserved for future checks).

    Returns:
        VerifierAuditResult with all detected findings.
    """
    findings: list[VerifierAuditFinding] = []
    total = len(results)

    # --- empty results ---
    if total == 0:
        findings.append(
            VerifierAuditFinding(
                severity="high",
                check_id="empty_results",
                description="No results were provided for this verifier.",
                evidence="total_results=0",
                recommendation=(
                    "Ensure the verifier is actually invoked and its results "
                    "are collected before calling audit_verifier_results."
                ),
            )
        )
        return VerifierAuditResult(
            verifier_id=verifier_id,
            total_results=0,
            pass_rate=0.0,
            findings=findings,
        )

    # Compute basic stats
    passed_count = sum(1 for r in results if r.get("passed"))
    pass_rate = passed_count / total

    scores: list[Any] = [r.get("score") for r in results]
    non_none_scores = [s for s in scores if s is not None]

    # --- oracle leakage risk (critical, checked early) ---
    lowered = verifier_id.lower()
    if "oracle" in lowered or "ground_truth" in lowered:
        findings.append(
            VerifierAuditFinding(
                severity="critical",
                check_id="oracle_leakage_risk",
                description=(
                    f"Verifier id '{verifier_id}' contains 'oracle' or "
                    "'ground_truth', indicating possible label leakage."
                ),
                evidence=f"verifier_id={verifier_id!r}",
                recommendation=(
                    "Rename the verifier and confirm it does not have direct "
                    "access to held-out labels during evaluation."
                ),
            )
        )

    # --- high pass-rate (high severity) ---
    if pass_rate > 0.95 and total >= 10:
        findings.append(
            VerifierAuditFinding(
                severity="high",
                check_id="high_pass_rate",
                description=(
                    f"Pass rate is {pass_rate:.1%} (>{0.95:.0%}) across {total} results, "
                    "suggesting the verifier may be too lenient or degenerate."
                ),
                evidence=f"pass_rate={pass_rate:.4f}, total={total}",
                recommendation=(
                    "Review verifier thresholds; consider adversarial test cases "
                    "to confirm the verifier rejects genuinely bad outputs."
                ),
            )
        )

    # --- no score signal (medium severity) ---
    if len(non_none_scores) == 0:
        findings.append(
            VerifierAuditFinding(
                severity="medium",
                check_id="no_score_signal",
                description=(
                    "All result scores are None; the verifier produces no "
                    "continuous signal beyond pass/fail."
                ),
                evidence=f"non_none_scores=0 out of {total}",
                recommendation=(
                    "Add a numeric confidence score to enable ranking, "
                    "calibration analysis, and downstream reward modelling."
                ),
            )
        )
    else:
        # --- binary score only (medium severity) ---
        unique_scores = set(non_none_scores)
        if len(unique_scores) <= 2:
            findings.append(
                VerifierAuditFinding(
                    severity="medium",
                    check_id="binary_score_only",
                    description=(
                        f"Only {len(unique_scores)} unique score value(s) observed "
                        f"({sorted(unique_scores)}); score is effectively binary."
                    ),
                    evidence=(
                        f"unique_score_values={sorted(unique_scores)}, "
                        f"non_none_count={len(non_none_scores)}"
                    ),
                    recommendation=(
                        "Return a continuous score in [0, 1] to provide richer "
                        "signal for reward modelling and result ranking."
                    ),
                )
            )

    # --- low pass-rate (medium severity) ---
    if pass_rate < 0.05 and total >= 10:
        findings.append(
            VerifierAuditFinding(
                severity="medium",
                check_id="low_pass_rate",
                description=(
                    f"Pass rate is {pass_rate:.1%} (<5%) across {total} results, "
                    "suggesting the verifier may be too strict or broken."
                ),
                evidence=f"pass_rate={pass_rate:.4f}, total={total}",
                recommendation=(
                    "Review verifier thresholds and confirm the task inputs are "
                    "representative; a near-zero pass rate may indicate a bug."
                ),
            )
        )

    # --- missing claim_ids (low severity) ---
    missing_claim_count = sum(
        1
        for r in results
        if not r.get("claim_ids")  # None, [], or missing key all count
    )
    if missing_claim_count > total * 0.5:
        findings.append(
            VerifierAuditFinding(
                severity="low",
                check_id="missing_claim_ids",
                description=(
                    f"{missing_claim_count}/{total} results ({missing_claim_count/total:.1%}) "
                    "are missing claim_ids, reducing traceability."
                ),
                evidence=(
                    f"missing_claim_ids_count={missing_claim_count}, total={total}"
                ),
                recommendation=(
                    "Populate claim_ids with the constraint/claim identifiers that "
                    "the verifier checked so failures can be linked back to the spec."
                ),
            )
        )

    return VerifierAuditResult(
        verifier_id=verifier_id,
        total_results=total,
        pass_rate=pass_rate,
        findings=findings,
    )


def audit_aep_verifiers(
    aep_records: Sequence[dict[str, Any]],
) -> dict[str, VerifierAuditResult]:
    """Audit all verifiers referenced across a collection of AEP records.

    Each AEP record is expected to contain a "verifier_results" key whose
    value is a list of dicts, each with at least a "verifier_id" field plus
    the standard "passed" / "score" / "claim_ids" fields consumed by
    audit_verifier_results.

    Args:
        aep_records: sequence of AEP record dicts (e.g. loaded from JSONL).

    Returns:
        Dict mapping verifier_id -> VerifierAuditResult.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record in aep_records:
        verifier_results = record.get("verifier_results") or []
        for vr in verifier_results:
            vid = vr.get("verifier_id", "<unknown>")
            grouped.setdefault(vid, []).append(vr)

    return {
        vid: audit_verifier_results(vid, vr_list)
        for vid, vr_list in grouped.items()
    }


__all__ = [
    "VerifierAuditFinding",
    "VerifierAuditResult",
    "audit_aep_verifiers",
    "audit_verifier_results",
]
