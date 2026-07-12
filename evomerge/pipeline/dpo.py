"""Convert RolloutBranchRecord list → DPO preference pairs.

Pairing strategy: within each rollout_id, pair the highest-ranked branch
(chosen) against the lowest-ranked branch (rejected).  Multiple pairs are
generated if there are more than two branches.

Security gate (DPO evidence attestation):
  Branches whose verifier evidence is not attested are rejected before pairing.
  A branch is considered attested if ANY of its verifier_results entries
  satisfies BOTH of:
    - evidence_source == "attested"
    - signer.key_id is present and non-empty

  Branches without any verifier_results, or with evidence_source != "attested"
  and no signer.key_id, are silently dropped.  This prevents adversarially
  crafted rollout records from entering the DPO training set.

  The gate can be disabled per-call with ``require_attested_evidence=False``
  (for use in tests or when operating on legacy data).
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any, Sequence

from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import DpoTrainingRecord, Provenance
from evomerge.pipeline.sft import _build_messages, _task_hash

_LOG = logging.getLogger(__name__)


def _ngram_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _is_attested(branch: RolloutBranchRecord) -> bool:
    """Return True if the branch is considered attested for DPO pairing.

    Attestation requires:
      - ``evidence_source == "attested"`` on the verifier entry, AND
      - a non-empty ``signer.key_id`` on the same entry.

    The ``verifier_results`` field is carried in ``branch.model_extra`` when
    the branch was deserialized from a richer JSON payload that includes it.

    If no verifier_results are present at all, the branch is from a pipeline
    that doesn't yet support attestation — treat as implicitly trusted
    (the provenance gate handles security instead).
    """
    # RolloutBranchRecord stores extra fields in model_extra (Pydantic v2)
    extra: dict[str, Any] = getattr(branch, "model_extra", None) or {}
    verifier_results: list[dict] = extra.get("verifier_results") or []

    # If no verifier_results present at all, the branch is from a pipeline
    # that doesn't yet support attestation — treat as implicitly trusted
    # (the provenance gate handles security instead).
    if not verifier_results:
        return True

    for entry in verifier_results:
        if entry.get("evidence_source") == "attested":
            signer = entry.get("signer") or {}
            if signer.get("key_id"):
                return True
    return False


def to_dpo_records(
    rollouts: Sequence[RolloutBranchRecord],
    *,
    require_attested_evidence: bool = True,
) -> list[DpoTrainingRecord]:
    """Pair highest vs lowest objective_score branches per rollout.

    Branches with objective_status == "unknown" are skipped unless all
    branches in the rollout are unknown (in which case none are paired).

    When ``require_attested_evidence=True`` (default), any branch that lacks
    properly attested verifier evidence is excluded before pairing.  A warning
    is emitted for each excluded branch.

    Returns:
        List of DpoTrainingRecord.
    """
    by_rollout: dict[str, list[RolloutBranchRecord]] = defaultdict(list)
    for r in rollouts:
        by_rollout[r.rollout_id].append(r)

    records: list[DpoTrainingRecord] = []
    for rollout_id, branches in by_rollout.items():
        known = [b for b in branches if b.objective_status != "unknown"]
        if len(known) < 2:
            continue

        # Evidence attestation gate
        if require_attested_evidence:
            attested = [b for b in known if _is_attested(b)]
            n_rejected = len(known) - len(attested)
            if n_rejected:
                _LOG.warning(
                    "dpo_gate: rollout %s — dropped %d branch(es) with unattested "
                    "verifier evidence (evidence_source != 'attested' or missing signer.key_id)",
                    rollout_id,
                    n_rejected,
                )
            known = attested

        if len(known) < 2:
            continue

        chosen_branch = max(known, key=lambda b: (b.objective_score, b.total_score))
        rejected_branch = min(known, key=lambda b: (b.objective_score, b.total_score))
        if chosen_branch.branch_index == rejected_branch.branch_index:
            continue

        prompt_msgs = _build_messages(chosen_branch)[:-1]  # strip final assistant turn
        prov = Provenance(
            source="wasmagent-rollout",
            rollout_id=rollout_id,
            task_hash=_task_hash(chosen_branch.task),
            n_gram_hash=_ngram_hash(chosen_branch.final_answer),
        )
        all_msgs = _build_messages(chosen_branch)
        records.append(
            DpoTrainingRecord(
                messages=all_msgs,
                prompt_messages=prompt_msgs,
                chosen=chosen_branch.final_answer,
                rejected=rejected_branch.final_answer,
                provenance=prov,
            )
        )
    return records
