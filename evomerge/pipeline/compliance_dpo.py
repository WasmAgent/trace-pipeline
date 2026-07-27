"""Convert ComplianceEvalRecord list → DPO preference pairs.

Implements plan Section 6 Phase 2: "用 deterministic verifier + teacher model
+ human review 共同构造偏好数据".

Pairing strategy:
  chosen  — the artifact that passed all constraints (final_pass=True, fewest
             repair rounds or zero violations)
  rejected — either:
    (a) an earlier draft that violated constraints (artifact before repair), or
    (b) a synthetic bad output injected by the caller via `bad_outputs`

When a ComplianceEvalRecord has repair_trace entries, each round that
transitioned from failing to passing can produce one chosen/rejected pair:
  chosen   = final compliant artifact
  rejected = what the model produced before the repair that resolved violations

If no repair trace is available (record passed on first attempt) and no
bad_outputs are provided, the record is skipped — a passing-only record
yields no preference signal.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from evomerge.schemas.compliance import ComplianceEvalRecord
from evomerge.schemas.training import DpoTrainingRecord, Message, Provenance


def _task_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def _ngram_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compliance_to_dpo_records(
    records: Sequence[ComplianceEvalRecord],
    *,
    bad_outputs: dict[str, list[str]] | None = None,
) -> list[DpoTrainingRecord]:
    """Convert compliance eval records to DPO preference pairs.

    Args:
        records: ComplianceEvalRecord list from the compliance engine.
        bad_outputs: optional dict mapping task_id → list of bad output strings
            (e.g. from a teacher model or from earlier failed runs). Each bad
            output is paired against the final compliant artifact as rejected.

    Returns:
        List of DpoTrainingRecord. Records with no pairing signal are skipped.

    Pairing logic:
        1. If repair_trace contains rounds with ok=True (repair succeeded):
           for each such round, produce a pair using the artifact as chosen and
           a reconstructed "before-repair" context as rejected input.
        2. If bad_outputs[task_id] is provided: pair each bad output (rejected)
           against the final compliant artifact (chosen).
        3. If record passed with zero repair rounds and no bad_outputs: skip.
        4. If record failed (final_pass=False): skip regardless.
    """
    result: list[DpoTrainingRecord] = []
    bad_outputs = bad_outputs or {}

    for rec in records:
        if not rec.final_pass:
            continue  # failed records have no "chosen" side

        prov = Provenance(
            source="wasmagent-compliance",
            task_id=rec.task_id,
            n_gram_hash=_ngram_hash(rec.artifact),
            task_hash=_task_hash(rec.task_id),
        )

        # Strategy 1: repair-trace pairs
        for entry in rec.repair_trace:
            if not entry.ok:
                continue
            # The repair resolved these violations — construct a rejected draft
            # that represents what the model produced before repair
            violation_hints = "\n".join(
                f"- {v.hint}"
                for v in rec.violations
                if v.constraint_id in entry.violation_ids
            )
            if not violation_hints:
                continue
            # rejected: pre-repair draft (we don't have it verbatim, so we
            # reconstruct a minimal representation that carries the violation context)
            rejected_repr = (
                f"[pre-repair draft — round {entry.round}]\n"
                f"Violations present:\n{violation_hints}\n\n"
                f"Draft artifact (incomplete):\n{rec.artifact}"
            )
            result.append(
                DpoTrainingRecord(
                    messages=[
                        Message(role="user", content=f"task_id={rec.task_id}"),
                        Message(role="assistant", content=rec.artifact),
                    ],
                    prompt_messages=[Message(role="user", content=f"task_id={rec.task_id}")],
                    chosen=rec.artifact,
                    rejected=rejected_repr,
                    loss_weight_tokens="recovery",
                    provenance=prov,
                )
            )

        # Strategy 2: explicit bad_outputs pairs
        for bad in bad_outputs.get(rec.task_id, []):
            if bad.strip() == rec.artifact.strip():
                continue  # identical — no preference signal
            result.append(
                DpoTrainingRecord(
                    messages=[
                        Message(role="user", content=f"task_id={rec.task_id}"),
                        Message(role="assistant", content=rec.artifact),
                    ],
                    prompt_messages=[Message(role="user", content=f"task_id={rec.task_id}")],
                    chosen=rec.artifact,
                    rejected=bad,
                    provenance=prov,
                )
            )

    return result
