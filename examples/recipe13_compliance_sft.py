"""recipe13_compliance_sft.py — convert ComplianceEvalRecord list to SFT records.

Shows how compliance engine output (TaskSpec + violations + repair trace)
becomes answerer and repairer training records.

Run:
    python examples/recipe13_compliance_sft.py
"""
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
from evomerge.pipeline.compliance_sft import compliance_to_sft_records

# --- build two synthetic compliance records ---
violation = ConstraintViolation(
    constraint_id="c_action_list",
    level=ConstraintLevel.hard,
    category=ConstraintCategory.content,
    hint="Output is missing the required 'Action List' section",
    detected_at=ViolationStage.post_decode,
)
repair_entry = RepairTraceEntry(
    round=1,
    violation_ids=["c_action_list"],
    strategy=RepairStrategy.insert_section,
    ok=True,
)

record_passed = ComplianceEvalRecord(
    task_id="rfi-001",
    task_spec_hash="abc123",
    model="qwen-7b",
    mode=RunMode.full_pcl,
    violations=[violation],
    repair_trace=[repair_entry],
    repair_rounds=1,
    final_pass=True,
    artifact="Executive Summary\n...\nAction List\n- Review proposal by Friday\n- Schedule kickoff",
)

record_failed = ComplianceEvalRecord(
    task_id="rfi-002",
    task_spec_hash="abc123",
    model="qwen-7b",
    mode=RunMode.full_pcl,
    violations=[violation],
    repair_trace=[],
    repair_rounds=0,
    final_pass=False,
    artifact="Executive Summary\n...",
)

# --- convert: default excludes failures ---
sft_default = compliance_to_sft_records([record_passed, record_failed])
sft_with_failures = compliance_to_sft_records(
    [record_passed, record_failed], include_failures=True
)

print(f"records (exclude failures): {len(sft_default)}")
print(f"records (include failures): {len(sft_with_failures)}")

for r in sft_default:
    print(f"\n  output_type    : {r.output_type!r}")
    print(f"  loss_weight    : {r.loss_weight_tokens!r}")
    print(f"  user content   : '{r.messages[0].content[:80]}'")
    print(f"  assistant      : '{r.messages[-1].content[:80]}'")

print("\noutput_types:", [r.output_type for r in sft_default])
assert "final_answer" in [r.output_type for r in sft_default]
assert "repair_patch" in [r.output_type for r in sft_default]
print("compliance SFT conversion OK")
