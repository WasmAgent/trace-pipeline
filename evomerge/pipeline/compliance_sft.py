"""Convert ComplianceEvalRecord list → SFT training records.

Each compliance record yields up to three record types depending on what
happened during the run:

  answerer  — TaskSpec + artifact → final_answer (always, when final_pass=True)
  repairer  — TaskSpec + bad_output + violation hint → repair_patch
              (one per repair round that succeeded, ok=True)
  tool_call — TaskSpec + allowed_tools → first tool call in the trace
              (when the record's artifact was driven by tool calls)

Records where final_pass=False and no repair succeeded are skipped unless
include_failures=True is passed (useful for DPO negative construction).
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from evomerge.schemas.compliance import ComplianceEvalRecord
from evomerge.schemas.training import Message, Provenance, SftTrainingRecord


def _task_context(record: ComplianceEvalRecord) -> str:
    """Serialise task_id + mode as a minimal system context."""
    return json.dumps(
        {"task_id": record.task_id, "mode": record.mode.value}, ensure_ascii=False
    )


def _violation_summary(record: ComplianceEvalRecord) -> str:
    if not record.violations:
        return ""
    lines = [
        f"- [{v.level.value}/{v.category.value}] {v.hint}"
        for v in record.violations
    ]
    return "Constraint violations:\n" + "\n".join(lines)


def compliance_to_sft_records(
    records: Sequence[ComplianceEvalRecord],
    *,
    include_failures: bool = False,
) -> list[SftTrainingRecord]:
    """Convert compliance eval records to SFT training records.

    Args:
        records: output from the wasmagent-js compliance engine.
        include_failures: include runs where final_pass=False and no
            repair succeeded. Useful for building DPO rejected examples.

    Returns:
        Flat list of SftTrainingRecord (multiple per ComplianceEvalRecord
        when repair rounds are present).
    """
    results: list[SftTrainingRecord] = []
    for rec in records:
        prov = Provenance(
            source="wasmagent-compliance",
            task_id=rec.task_id,
        )

        # answerer record
        if rec.final_pass:
            violation_ctx = _violation_summary(rec)
            user_content = _task_context(rec)
            if violation_ctx:
                user_content = f"{user_content}\n\n{violation_ctx}"
            results.append(
                SftTrainingRecord(
                    messages=[
                        Message(role="user", content=user_content),
                        Message(role="assistant", content=rec.artifact),
                    ],
                    output_type="final_answer",
                    provenance=prov,
                )
            )
        elif include_failures:
            results.append(
                SftTrainingRecord(
                    messages=[
                        Message(role="user", content=_task_context(rec)),
                        Message(role="assistant", content=rec.artifact),
                    ],
                    output_type="final_answer",
                    loss_weight_tokens="recovery",
                    provenance=prov,
                )
            )

        # repairer records — one per successful repair round
        for entry in rec.repair_trace:
            if not entry.ok:
                continue
            violation_ids = entry.violation_ids
            relevant = [v for v in rec.violations if v.constraint_id in violation_ids]
            if not relevant:
                continue
            hint_block = "\n".join(
                f"- [{v.level.value}] {v.hint}" for v in relevant
            )
            results.append(
                SftTrainingRecord(
                    messages=[
                        Message(
                            role="user",
                            content=(
                                f"{_task_context(rec)}\n\n"
                                f"Round {entry.round} violations:\n{hint_block}"
                            ),
                        ),
                        Message(role="assistant", content=rec.artifact),
                    ],
                    output_type="repair_patch",
                    loss_weight_tokens="recovery",
                    provenance=prov,
                )
            )

    return results
