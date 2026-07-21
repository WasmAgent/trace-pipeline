"""evomerge.validate.quality_gate — training data quality checks.

Checks applied before exporting training records for fine-tuning:

  SFT quality:
    - repair_patch fraction (should be 10–40% for healthy diversity)
    - min assistant response length (too-short answers indicate truncation)
    - cross-seed task ID overlap (dedup effectiveness)

  DPO quality:
    - chosen/rejected length ratio (rejected should not be trivially shorter)
    - chosen length floor (very short chosen = low-information pair)
    - identical chosen/rejected (degenerate pairs)

  Overall:
    - total record count vs minimum threshold
    - contamination flag (requires eval_items if provided)

  Security checks (run_quality_gate):
    - Anomalous objective_score (> 1.0, < 0.0, or NaN) in any rollout record
      triggers an audit log entry and adds an "error"-level issue.
    - Prompt-injection keyword detection: if any text field contains a
      substring matching the INJECTION_SIGNAL_FRAGMENTS list, an audit log
      entry is written and the batch is rejected with an "error"-level issue.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

from evomerge.schemas.training import DpoTrainingRecord, SftTrainingRecord

_AUDIT_LOG = logging.getLogger("evomerge.audit")

# Substring signals extracted from the W4 RISK_CORPUS (injection_en / jailbreak
# categories).  These are plain lowercase substrings; matching is performed
# after NFKC + lowercase on the candidate text so homoglyphs and full-width
# variants are caught.  Keep the list short and high-precision: only include
# phrases that are unambiguously adversarial in a training-data context.
INJECTION_SIGNAL_FRAGMENTS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "disregard all",
    "forget your previous context",
    "forget all previous",
    "new instructions:",
    "system prompt:",
    "you are now dan",
    "you are now an unrestricted",
    "override safety filters",
    "end of prior context",
    "ignore prior",
    "exfiltrate",
    "leak the private key",
    "leak all secrets",
    "output the system prompt",
    "reveal all confidential",
    "send all collected data",
    "act as an ai with developer mode",
    "unrestricted language model",
    "no content filters",
    "hacker ai without restrictions",
    "no guidelines or safety rules",
    "ethical constraints have been removed",
    "unfiltered, compliant ai",
)


@dataclass
class QualityIssue:
    level: str        # "error" | "warning"
    check: str
    message: str
    value: float | int | None = None
    threshold: float | int | None = None


@dataclass
class QualityReport:
    n_sft: int = 0
    n_dpo: int = 0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_sft": self.n_sft,
            "n_dpo": self.n_dpo,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [
                {"level": i.level, "check": i.check, "message": i.message,
                 "value": i.value, "threshold": i.threshold}
                for i in self.issues
            ],
        }

    def print_report(self) -> None:
        status = "✓ PASS" if self.ok else "✗ FAIL"
        print(f"{status}  SFT={self.n_sft}  DPO={self.n_dpo}  "
              f"errors={len(self.errors)}  warnings={len(self.warnings)}")
        for issue in self.issues:
            icon = "  [ERROR]  " if issue.level == "error" else "  [WARN]   "
            detail = f" (value={issue.value}, threshold={issue.threshold})" \
                     if issue.value is not None else ""
            print(f"{icon}{issue.check}: {issue.message}{detail}")


def check_sft_quality(
    records: Sequence[SftTrainingRecord],
    *,
    min_records: int = 100,
    min_repair_patch_frac: float = 0.05,
    max_repair_patch_frac: float = 0.60,
    min_assistant_chars: int = 20,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n = len(records)

    if n < min_records:
        issues.append(QualityIssue(
            level="error", check="sft_min_records",
            message=f"only {n} SFT records, need ≥{min_records}",
            value=n, threshold=min_records,
        ))

    if n == 0:
        return issues

    output_types = {}
    for r in records:
        t = r.output_type
        output_types[t] = output_types.get(t, 0) + 1

    n_repair = output_types.get("repair_patch", 0)
    repair_frac = n_repair / n

    if repair_frac < min_repair_patch_frac:
        issues.append(QualityIssue(
            level="warning", check="sft_repair_patch_fraction",
            message=f"repair_patch fraction {repair_frac:.1%} below minimum {min_repair_patch_frac:.1%} — diversity may be low",
            value=round(repair_frac, 4), threshold=min_repair_patch_frac,
        ))
    if repair_frac > max_repair_patch_frac:
        issues.append(QualityIssue(
            level="warning", check="sft_repair_patch_fraction",
            message=f"repair_patch fraction {repair_frac:.1%} above maximum {max_repair_patch_frac:.1%} — may overfit repair path",
            value=round(repair_frac, 4), threshold=max_repair_patch_frac,
        ))

    short_count = 0
    for r in records:
        assistant_turns = [m for m in r.messages if m.role == "assistant"]
        if assistant_turns and len(assistant_turns[-1].content) < min_assistant_chars:
            short_count += 1
    if short_count > 0:
        issues.append(QualityIssue(
            level="warning", check="sft_short_assistant",
            message=f"{short_count}/{n} records have assistant response < {min_assistant_chars} chars (possible truncation)",
            value=short_count, threshold=0,
        ))

    return issues


def check_dpo_quality(
    records: Sequence[DpoTrainingRecord],
    *,
    min_records: int = 20,
    min_chosen_chars: int = 30,
    max_length_ratio: float = 10.0,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n = len(records)

    if n < min_records:
        issues.append(QualityIssue(
            level="warning", check="dpo_min_records",
            message=f"only {n} DPO pairs, recommended ≥{min_records}",
            value=n, threshold=min_records,
        ))

    if n == 0:
        return issues

    identical = sum(1 for r in records if r.chosen.strip() == r.rejected.strip())
    if identical > 0:
        issues.append(QualityIssue(
            level="error", check="dpo_identical_pairs",
            message=f"{identical} DPO pairs have identical chosen/rejected (degenerate — no learning signal)",
            value=identical, threshold=0,
        ))

    short_chosen = sum(1 for r in records if len(r.chosen) < min_chosen_chars)
    if short_chosen > 0:
        issues.append(QualityIssue(
            level="warning", check="dpo_short_chosen",
            message=f"{short_chosen}/{n} pairs have chosen < {min_chosen_chars} chars (low-information)",
            value=short_chosen, threshold=0,
        ))

    extreme_ratio = 0
    for r in records:
        lc, lr = len(r.chosen), len(r.rejected)
        if lr > 0 and lc / lr > max_length_ratio:
            extreme_ratio += 1
        elif lc > 0 and lr / lc > max_length_ratio:
            extreme_ratio += 1
    if extreme_ratio > 0:
        issues.append(QualityIssue(
            level="warning", check="dpo_extreme_length_ratio",
            message=f"{extreme_ratio}/{n} pairs have chosen/rejected length ratio > {max_length_ratio}x (spurious length bias)",
            value=extreme_ratio, threshold=0,
        ))

    return issues


def _nfkc_lower(text: str) -> str:
    """NFKC-normalise and lowercase; used for injection signal matching."""
    import unicodedata
    return unicodedata.normalize("NFKC", text).lower()


def check_anomalous_scores(
    rollout_records: Sequence[dict],
) -> list[QualityIssue]:
    """Check rollout dicts for anomalous objective_score values.

    A score is anomalous when it is NaN, infinite, > 1.0, or < 0.0.
    Each anomalous record is written to the audit log and the function
    returns one aggregated QualityIssue at "error" level.

    Args:
        rollout_records: raw dicts (e.g. from JSON deserialization) that
            should have an ``objective_score`` field.

    Returns:
        List of QualityIssue (at most one, aggregated).
    """
    issues: list[QualityIssue] = []
    anomalous: list[int] = []
    for idx, rec in enumerate(rollout_records):
        score = rec.get("objective_score")
        if score is None:
            continue
        try:
            fval = float(score)
        except (TypeError, ValueError):
            fval = float("nan")
        if math.isnan(fval) or math.isinf(fval) or fval < 0.0 or fval > 1.0:
            _AUDIT_LOG.warning(
                "anomalous_score: record index=%d objective_score=%r — "
                "expected value in [0.0, 1.0]; record excluded from training",
                idx,
                score,
            )
            anomalous.append(idx)
    if anomalous:
        issues.append(QualityIssue(
            level="error",
            check="anomalous_objective_score",
            message=(
                f"{len(anomalous)} record(s) have objective_score outside [0.0, 1.0] "
                f"or NaN/Inf (indices: {anomalous[:10]}{'...' if len(anomalous) > 10 else ''})"
            ),
            value=len(anomalous),
            threshold=0,
        ))
    return issues


def check_injection_signals(
    texts: Sequence[str],
    *,
    fragments: tuple[str, ...] = INJECTION_SIGNAL_FRAGMENTS,
) -> list[QualityIssue]:
    """Scan text fields for prompt-injection keyword signals.

    Each text is NFKC-normalised and lowercased before matching so that
    full-width homoglyphs and look-alike substitutions are caught.  Matching
    is substring-based (not regex) for efficiency and predictability.

    A hit is written to the audit log.  The function returns one aggregated
    QualityIssue at "error" level if any signal is found.

    Args:
        texts: iterable of strings to scan (e.g. task, final_answer fields).
        fragments: tuple of lowercase substring signals to match against.

    Returns:
        List of QualityIssue (at most one, aggregated).
    """
    issues: list[QualityIssue] = []
    hits: list[dict] = []
    for idx, raw in enumerate(texts):
        normalised = _nfkc_lower(raw)
        for frag in fragments:
            if frag in normalised:
                _AUDIT_LOG.warning(
                    "injection_signal: text index=%d matched fragment=%r — "
                    "record excluded; audit review required",
                    idx,
                    frag,
                )
                hits.append({"index": idx, "fragment": frag})
                break  # one hit per text is sufficient
    if hits:
        issues.append(QualityIssue(
            level="error",
            check="injection_signal",
            message=(
                f"{len(hits)} text(s) contain prompt-injection signal fragments — "
                f"review audit log; first hit: {hits[0]['fragment']!r}"
            ),
            value=len(hits),
            threshold=0,
        ))
    return issues


def run_quality_gate(
    sft_records: Sequence[SftTrainingRecord] | None = None,
    dpo_records: Sequence[DpoTrainingRecord] | None = None,
    *,
    eval_texts: Sequence[str] | None = None,
    contamination_threshold: float = 0.2,
    sft_min_records: int = 100,
    dpo_min_records: int = 20,
    rollout_records: Sequence[dict] | None = None,
) -> QualityReport:
    """Run all quality checks and return a QualityReport.

    Args:
        sft_records: SFT training records to check.
        dpo_records: DPO preference pairs to check.
        eval_texts: optional eval item texts for contamination check.
        contamination_threshold: Jaccard threshold for flagging contamination.
        sft_min_records: minimum SFT records required.
        dpo_min_records: minimum DPO records recommended.
        rollout_records: optional raw rollout dicts for anomalous-score and
            injection-signal checks.  When supplied, objective_score values are
            validated and text fields are scanned for injection signals.

    Returns:
        QualityReport — call .ok to check pass/fail, .print_report() for summary.
    """
    sft = list(sft_records) if sft_records else []
    dpo = list(dpo_records) if dpo_records else []

    report = QualityReport(n_sft=len(sft), n_dpo=len(dpo))

    if sft:
        report.issues.extend(check_sft_quality(sft, min_records=sft_min_records))

    if dpo:
        report.issues.extend(check_dpo_quality(dpo, min_records=dpo_min_records))

    # Anomalous score + injection signal checks on raw rollout dicts
    if rollout_records:
        report.issues.extend(check_anomalous_scores(rollout_records))
        # Collect all text fields for injection scanning
        injection_texts: list[str] = []
        for rec in rollout_records:
            for field_name in ("task", "final_answer"):
                val = rec.get(field_name)
                if isinstance(val, str):
                    injection_texts.append(val)
        if injection_texts:
            report.issues.extend(check_injection_signals(injection_texts))

    if eval_texts and (sft or dpo):
        from evomerge.validate.contamination import check_contamination
        outputs = (
            [r.messages[-1].content for r in sft if r.messages]
            + [r.chosen for r in dpo]
        )
        cont = check_contamination(outputs, list(eval_texts),
                                   threshold=contamination_threshold)
        if cont.n_flagged > 0:
            report.issues.append(QualityIssue(
                level="error", check="contamination",
                message=f"{cont.n_flagged}/{cont.n_training} records flagged for eval-set contamination (threshold={contamination_threshold})",
                value=cont.n_flagged, threshold=0,
            ))

    return report


__all__ = [
    "QualityIssue", "QualityReport", "check_dpo_quality",
    "check_sft_quality", "run_quality_gate",
    "check_anomalous_scores", "check_injection_signals",
    "INJECTION_SIGNAL_FRAGMENTS",
    "ADMISSION_CATEGORIES", "compute_admission_score", "admission_gate",
]


# ── Evidence Admission Score ──────────────────────────────────────────────────

ADMISSION_CATEGORIES = ("train_sft", "train_dpo", "train_repair", "train_router", "audit_only", "reject")


def compute_admission_score(
    aep_record: dict,
    quality_report: "QualityReport | None" = None,
) -> dict:
    """Compute an evidence admission score for a single AEP record.

    Returns a dict with:
      - score: float in [0.0, 1.0]
      - category: one of ADMISSION_CATEGORIES
      - reasons: list of str explaining the score
      - dimensions: dict of individual dimension scores
    """
    reasons: list[str] = []
    dims: dict[str, float] = {}

    # 1. Schema validity
    schema_ver = aep_record.get("schema_version", "")
    if schema_ver not in ("aep/v0.1", "aep/v0.2", "aep/v0.3", "aep/v0.4"):
        return {"score": 0.0, "category": "reject", "reasons": ["invalid schema_version"], "dimensions": {}}
    dims["schema_validity"] = 1.0

    # 2. Evidence completeness (state-changing actions with evidence)
    actions = aep_record.get("actions", [])
    state_changing = [a for a in actions if a.get("state_changing", False)]
    with_evidence = [a for a in state_changing if a.get("result_digest") or a.get("precondition_digest")]
    completeness = (len(with_evidence) / len(state_changing)) if state_changing else 1.0
    dims["evidence_completeness"] = completeness

    # 3. Policy compliance (capability_decisions present)
    decisions = aep_record.get("capability_decisions", [])
    has_policy = len(decisions) > 0 or len(state_changing) == 0
    dims["policy_compliance"] = 1.0 if has_policy else 0.5
    if not has_policy:
        reasons.append("no capability_decisions for state-changing run")

    # 4. Verifier strength (verifier_results present)
    verifiers = aep_record.get("verifier_results", [])
    verifier_score = min(1.0, len(verifiers) * 0.5) if verifiers else 0.0
    dims["verifier_strength"] = verifier_score
    if not verifiers:
        reasons.append("no verifier_results")

    # 5. Provenance (repo_commit + model_id)
    has_provenance = bool(aep_record.get("repo_commit")) and bool(aep_record.get("model_id"))
    dims["provenance"] = 1.0 if has_provenance else 0.5
    if not has_provenance:
        reasons.append("missing repo_commit or model_id")

    # 6. Contamination proxy (tool_manifest_digest present)
    has_digest = bool(aep_record.get("tool_manifest_digest"))
    dims["contamination_risk"] = 1.0 if has_digest else 0.7
    if not has_digest:
        reasons.append("missing tool_manifest_digest — contamination risk elevated")

    # 7. Recording mode bonus (v0.3+)
    recording_mode = aep_record.get("recording_mode")
    recording_mode_bonus = 0.0
    if recording_mode == "full":
        recording_mode_bonus = 0.05
    elif recording_mode == "delta":
        recording_mode_bonus = 0.02
    # "validation" or absent/null → no bonus

    # Weighted score
    weights = {
        "schema_validity": 0.20,
        "evidence_completeness": 0.25,
        "policy_compliance": 0.20,
        "verifier_strength": 0.15,
        "provenance": 0.10,
        "contamination_risk": 0.10,
    }
    score = sum(dims.get(k, 0.0) * w for k, w in weights.items())
    score = min(1.0, score + recording_mode_bonus)

    # Routing
    if score < 0.4:
        category = "reject"
        reasons.append(f"score {score:.2f} below reject threshold 0.40")
    elif score < 0.6:
        category = "audit_only"
    else:
        # Determine training category from verifier results and repair presence
        has_repair = any("repair" in str(v.get("verifier_id", "")).lower() for v in verifiers)
        has_dpo_pair = aep_record.get("_dpo_pair_id")  # set by evomerge pipeline
        passed_count = sum(1 for v in verifiers if v.get("passed"))
        if has_repair:
            category = "train_repair"
        elif has_dpo_pair:
            category = "train_dpo"
        elif passed_count > 0 and completeness >= 0.8:
            category = "train_sft"
        else:
            category = "train_router"

    return {
        "score": round(score, 4),
        "category": category,
        "reasons": reasons,
        "dimensions": {k: round(v, 4) for k, v in dims.items()},
    }


def admission_gate(
    aep_records: list[dict],
    min_score: float = 0.6,
) -> dict:
    """Run admission scoring over a list of AEP records.

    Returns summary with per-category counts and list of scored records.
    """
    scored = [
        {"record": r, **compute_admission_score(r)}
        for r in aep_records
    ]
    by_category: dict[str, int] = {c: 0 for c in ADMISSION_CATEGORIES}
    for s in scored:
        by_category[s["category"]] = by_category.get(s["category"], 0) + 1

    return {
        "total": len(scored),
        "by_category": by_category,
        "admitted": [s for s in scored if s["category"] not in ("reject", "audit_only")],
        "rejected": [s for s in scored if s["category"] == "reject"],
        "audit_only": [s for s in scored if s["category"] == "audit_only"],
        "mean_score": round(sum(s["score"] for s in scored) / len(scored), 4) if scored else 0.0,
    }
