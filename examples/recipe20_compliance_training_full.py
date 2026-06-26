"""recipe20_compliance_training_full.py — full compliance-conditioned training pipeline.

Demonstrates the end-to-end path from ComplianceEvalRecords to routed training
data (SFT + DPO) using Evidence Admission Score filtering.

Steps:
  1. Build 10 synthetic ComplianceEvalRecord objects (6 pass, 4 fail).
  2. Convert them to AEP-like dicts for the admission gate.
  3. Run admission_gate to score and route each record.
  4. Extract passing ComplianceEvalRecords and route to SFT vs DPO.
  5. Print a structured summary with admission statistics.

Run:
    python examples/recipe20_compliance_training_full.py
"""
from __future__ import annotations

from evomerge.schemas.compliance import (
    ComplianceEvalRecord,
    ConstraintCategory,
    ConstraintLevel,
    ConstraintViolation,
    RepairStrategy,
    RepairTraceEntry,
    RunMode,
    ViolationStage,
)
from evomerge.pipeline.compliance_dpo import compliance_to_dpo_records
from evomerge.pipeline.compliance_sft import compliance_to_sft_records
from evomerge.validate.quality_gate import admission_gate


# ── 1. Build synthetic ComplianceEvalRecords ──────────────────────────────────

_VIOLATION = ConstraintViolation(
    constraint_id="c_section_required",
    level=ConstraintLevel.hard,
    category=ConstraintCategory.content,
    hint="Output is missing the required 'Summary' section",
    detected_at=ViolationStage.post_decode,
)

_REPAIR_ENTRY = RepairTraceEntry(
    round=1,
    violation_ids=["c_section_required"],
    strategy=RepairStrategy.insert_section,
    ok=True,
)


def _make_record(
    task_id: str,
    final_pass: bool,
    has_repair: bool = False,
) -> ComplianceEvalRecord:
    artifact = (
        "Summary\nThis proposal covers the rollout plan.\n\nAction List\n- Step A\n- Step B"
        if final_pass
        else "This proposal covers the rollout plan."
    )
    return ComplianceEvalRecord(
        task_id=task_id,
        task_spec_hash="synth-hash-001",
        model="qwen2.5-7b",
        mode=RunMode.full_pcl,
        violations=[_VIOLATION],
        repair_trace=[_REPAIR_ENTRY] if has_repair else [],
        repair_rounds=1 if has_repair else 0,
        final_pass=final_pass,
        artifact=artifact,
    )


# 6 passing (some with repair trace for DPO signal), 4 failing
compliance_records: list[ComplianceEvalRecord] = [
    _make_record("task-001", final_pass=True, has_repair=True),
    _make_record("task-002", final_pass=True, has_repair=True),
    _make_record("task-003", final_pass=True, has_repair=True),
    _make_record("task-004", final_pass=True, has_repair=False),
    _make_record("task-005", final_pass=True, has_repair=False),
    _make_record("task-006", final_pass=True, has_repair=False),
    _make_record("task-007", final_pass=False),
    _make_record("task-008", final_pass=False),
    _make_record("task-009", final_pass=False),
    _make_record("task-010", final_pass=False),
]

print(f"synthetic records: {len(compliance_records)}  "
      f"(pass={sum(1 for r in compliance_records if r.final_pass)}, "
      f"fail={sum(1 for r in compliance_records if not r.final_pass)})")


# ── 2. Convert to AEP-like dicts for admission gate ──────────────────────────
#
# The admission gate expects AEP-envelope dicts.  We synthesise minimal envelopes
# that carry the compliance outcome in verifier_results and preserve enough
# metadata for the scoring dimensions (schema_version, model_id, repo_commit,
# tool_manifest_digest, verifier_results, capability_decisions).

def _compliance_to_aep(rec: ComplianceEvalRecord) -> dict:
    """Map a ComplianceEvalRecord to a minimal AEP-v0.2 envelope."""
    verifier_results = [
        {
            "verifier_id": "compliance/deterministic",
            "passed": rec.final_pass,
            "violation_count": len(rec.violations),
        }
    ]
    if rec.repair_trace:
        verifier_results.append(
            {
                "verifier_id": "compliance/repair_verifier",
                "passed": rec.final_pass,
                "repair_rounds": rec.repair_rounds,
            }
        )

    return {
        "schema_version": "aep/v0.2",
        "task_id": rec.task_id,
        "model_id": rec.model,
        "repo_commit": "synth-commit-abc123",
        "tool_manifest_digest": "sha256:synth-manifest-digest",
        "actions": [],
        "capability_decisions": [{"policy": "allow", "tool": "compliance_check"}],
        "verifier_results": verifier_results,
        # tag for DPO routing in admission gate
        "_dpo_pair_id": rec.task_id if rec.repair_trace else None,
        # carry original record for downstream routing
        "_compliance_record": rec,
    }


aep_envelopes = [_compliance_to_aep(r) for r in compliance_records]


# ── 3. Run admission gate ─────────────────────────────────────────────────────

gate_result = admission_gate(aep_envelopes, min_score=0.6)

print("\n--- Admission Gate Results ---")
print(f"total records  : {gate_result['total']}")
print(f"mean score     : {gate_result['mean_score']:.4f}")
print(f"admitted       : {len(gate_result['admitted'])}")
print(f"audit_only     : {len(gate_result['audit_only'])}")
print(f"rejected       : {len(gate_result['rejected'])}")
print("\nby category:")
for cat, count in sorted(gate_result["by_category"].items()):
    print(f"  {cat:<20} {count}")


# ── 4. Route admitted records to SFT and DPO ─────────────────────────────────

admitted_compliance: list[ComplianceEvalRecord] = [
    s["record"]["_compliance_record"]
    for s in gate_result["admitted"]
]

sft_records = compliance_to_sft_records(admitted_compliance)
dpo_records = compliance_to_dpo_records(admitted_compliance)


# ── 5. Summary ────────────────────────────────────────────────────────────────

print("\n--- Training Data Summary ---")
print(f"admitted compliance records : {len(admitted_compliance)}")
print(f"SFT records produced        : {len(sft_records)}")
print(f"DPO pairs produced          : {len(dpo_records)}")

if sft_records:
    output_types: dict[str, int] = {}
    for r in sft_records:
        output_types[r.output_type] = output_types.get(r.output_type, 0) + 1
    print(f"SFT output_type breakdown   : {output_types}")

if dpo_records:
    print(f"DPO loss_weight_tokens      : "
          f"{[r.loss_weight_tokens for r in dpo_records]}")

print("\n--- Admission Statistics by Pass/Fail ---")
n_pass_admitted = sum(
    1 for s in gate_result["admitted"]
    if s["record"]["_compliance_record"].final_pass
)
n_fail_admitted = sum(
    1 for s in gate_result["admitted"]
    if not s["record"]["_compliance_record"].final_pass
)
print(f"admitted passing records : {n_pass_admitted}")
print(f"admitted failing records : {n_fail_admitted}")
print(f"audit_only records       : {len(gate_result['audit_only'])}")
print(f"rejected records         : {len(gate_result['rejected'])}")

assert len(compliance_records) == 10, "expected 10 synthetic records"
assert gate_result["total"] == 10, "gate should score all 10"
print("\ncompliance training pipeline OK")
