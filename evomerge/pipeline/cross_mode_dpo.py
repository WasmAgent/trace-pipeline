"""Cross-mode DPO pair construction from ComplianceEvalRecord triples.

When the same task_id has been run under multiple RunModes (direct,
prompt_retry, full_pcl), their relative pass/fail outcomes define
a natural preference ordering:

  full_pcl wins over direct      → strong signal (5+ constraint fixes)
  full_pcl wins over prompt_retry → medium signal (repair > retry)
  prompt_retry wins over full_pcl → boundary case (retry sometimes better)

This converter groups records by task_id and emits DPO pairs for every
mode pair where one passes and the other fails.  Records where both pass
or both fail produce no pair (no preference signal).

Typical usage:

    from evomerge.pipeline.cross_mode_dpo import cross_mode_dpo_records
    from evomerge.io import load_compliance_records, write_jsonl

    records = load_compliance_records("benchmarks/ifeval/results/runs.jsonl")
    pairs = cross_mode_dpo_records(records)
    write_jsonl(pairs, "data/training/cross_mode_dpo.jsonl")
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from evomerge.schemas.compliance import ComplianceEvalRecord, RunMode
from evomerge.schemas.training import DpoTrainingRecord, Message, Provenance

# Priority ordering: higher index = stronger mode
_MODE_RANK: dict[str, int] = {
    RunMode.direct: 0,
    RunMode.prompt_retry: 1,
    RunMode.full_pcl: 2,
}


def _task_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def _ngram_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _violation_summary(rec: ComplianceEvalRecord) -> str:
    if not rec.violations:
        return ""
    lines = [f"- [{v.level.value}/{v.category.value}] {v.hint}"
             for v in rec.violations]
    return "Violations:\n" + "\n".join(lines)


def cross_mode_dpo_records(
    records: Sequence[ComplianceEvalRecord],
    *,
    include_boundary_cases: bool = True,
) -> list[DpoTrainingRecord]:
    """Build DPO pairs from same-task, different-mode ComplianceEvalRecords.

    Args:
        records: flat list of ComplianceEvalRecord (multiple modes per task_id).
        include_boundary_cases: if True (default), also include pairs where a
            lower-ranked mode (e.g. prompt_retry) beat a higher-ranked mode
            (full_pcl).  These are valuable negative-repair training examples.

    Returns:
        List of DpoTrainingRecord.  One record per (task_id, mode_a, mode_b)
        pair where one passed and the other failed.
    """
    # Group by (task_id, model) — same task on different models should not pair
    by_key: dict[tuple[str, str], dict[str, ComplianceEvalRecord]] = defaultdict(dict)
    for rec in records:
        key = (rec.task_id, rec.model)
        by_key[key][rec.mode.value] = rec

    result: list[DpoTrainingRecord] = []

    for (task_id, model), mode_map in by_key.items():
        modes = list(mode_map.keys())
        # Compare every pair of modes
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i + 1:]:
                rec_a = mode_map[mode_a]
                rec_b = mode_map[mode_b]

                passes_a = rec_a.final_pass
                passes_b = rec_b.final_pass

                # Both pass or both fail → no preference signal
                if passes_a == passes_b:
                    continue

                # Determine chosen / rejected
                if passes_a and not passes_b:
                    chosen_rec, rejected_rec = rec_a, rec_b
                else:
                    chosen_rec, rejected_rec = rec_b, rec_a

                # Boundary case: lower-ranked mode beat higher-ranked
                rank_chosen   = _MODE_RANK.get(chosen_rec.mode, -1)
                rank_rejected = _MODE_RANK.get(rejected_rec.mode, -1)
                is_boundary   = rank_chosen < rank_rejected
                if is_boundary and not include_boundary_cases:
                    continue

                prov = Provenance(
                    source="wasmagent-compliance-cross-mode",
                    task_id=task_id,
                    n_gram_hash=_ngram_hash(chosen_rec.artifact),
                    task_hash=_task_hash(task_id),
                )

                # Build user context: task spec hash + violation summary of rejected
                violation_ctx = _violation_summary(rejected_rec)
                user_content = (
                    f"task_id={task_id} model={model}\n"
                    f"task_spec_hash={chosen_rec.task_spec_hash[:16]}\n"
                    + (f"\n{violation_ctx}" if violation_ctx else "")
                )

                result.append(
                    DpoTrainingRecord(
                        messages=[
                            Message(role="user", content=user_content),
                            Message(role="assistant", content=chosen_rec.artifact),
                        ],
                        prompt_messages=[
                            Message(role="user", content=user_content),
                        ],
                        chosen=chosen_rec.artifact,
                        rejected=rejected_rec.artifact,
                        loss_weight_tokens=(
                            "recovery" if chosen_rec.repair_rounds > 0 else "default"
                        ),
                        provenance=prov,
                    )
                )

    return result


def cross_mode_summary(
    records: Sequence[ComplianceEvalRecord],
) -> dict:
    """Return pairing statistics without generating records.

    Useful for a quick audit before committing to export.
    """
    by_key: dict[tuple[str, str], dict[str, ComplianceEvalRecord]] = defaultdict(dict)
    for rec in records:
        by_key[(rec.task_id, rec.model)][rec.mode.value] = rec

    stats: dict[str, int] = {
        "tasks_with_all_3_modes": 0,
        "pcl_beats_direct": 0,
        "pcl_beats_retry": 0,
        "retry_beats_pcl": 0,
        "direct_beats_retry": 0,
        "all_pass_no_signal": 0,
        "all_fail_no_signal": 0,
        "only_pcl_passes": 0,
    }

    for (task_id, model), mode_map in by_key.items():
        d  = mode_map.get("direct")
        pr = mode_map.get("prompt_retry")
        p  = mode_map.get("full_pcl")

        if d and pr and p:
            stats["tasks_with_all_3_modes"] += 1
            dp, prp, pp = d.final_pass, pr.final_pass, p.final_pass
            if pp and not dp:  stats["pcl_beats_direct"] += 1
            if pp and not prp: stats["pcl_beats_retry"]  += 1
            if prp and not pp: stats["retry_beats_pcl"]  += 1
            if dp and not prp and not pp: stats["direct_beats_retry"] += 1
            if dp and prp and pp:  stats["all_pass_no_signal"] += 1
            if not dp and not prp and not pp: stats["all_fail_no_signal"] += 1
            if pp and not dp and not prp: stats["only_pcl_passes"] += 1

    return stats


__all__ = ["cross_mode_dpo_records", "cross_mode_summary"]
