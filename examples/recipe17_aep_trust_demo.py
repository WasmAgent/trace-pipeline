"""recipe17_aep_trust_demo.py — AEP evidence pipeline end-to-end demo.

Demonstrates the complete WasmAgent evidence pipeline:

  synthetic agent run
    → AEP record (aep/v0.1)
    → AEP validation
    → AgentTrustScore (7 dimensions)
    → exploit surface classification
    → benchmark audit report (Markdown)
    → RunReceipt (SCITT-style provenance)
    → Evidence Registry (policy_bundle + receipt + verifier)

No model, no API key, no external downloads required.

Run:
    python examples/recipe17_aep_trust_demo.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# ── Step 1: Synthetic AEP record ─────────────────────────────────────────────
print("=" * 60)
print("Step 1: Build synthetic AEP record (aep/v0.1)")
print("=" * 60)

AEP_RECORD = {
    "schema_version": "aep/v0.1",
    "run_id": "recipe17-demo-001",
    "trace_id": "tr-recipe17",
    "model_id": "qwen2.5-1.5b-instruct",
    "model_provider": "local",
    "input_refs": [{"uri": "bscode/task/bscode-worker-api-001"}],
    "output_refs": [{"uri": "bscode/result/recipe17-demo-001", "redaction_profile": "bscode/pii-redact/v1"}],
    "capability_decisions": [
        {"capability": "tool:read_file", "subject": "agent", "resource": "src/routes/sessions.ts", "decision": "allow"},
        {"capability": "tool:write_file", "subject": "agent", "resource": "src/routes/sessions.ts", "decision": "allow"},
    ],
    "actions": [
        {
            "action_id": "action-0",
            "tool_name": "read_file",
            "state_changing": False,
            "evidence_refs": ["oai-span/s3"],
            "timestamp_ms": 1000.0,
        },
        {
            "action_id": "action-1",
            "tool_name": "write_file",
            "state_changing": True,
            "result_digest": "a" * 64,
            "evidence_refs": ["oai-span/s4"],
            "timestamp_ms": 2000.0,
        },
    ],
    "verifier_results": [
        {"verifier_id": "build-passes", "passed": True, "score": 1.0, "claim_ids": ["build-ok"]},
        {"verifier_id": "diff-matches-spec", "passed": True, "score": 1.0, "claim_ids": ["diff-ok"]},
    ],
    "budget_ledger": {
        "token_budget": {"limit": 4096, "spent": 1823},
        "tool_budget": {"limit": 10, "spent": 2},
    },
    "created_at_ms": 1000.0,
}

print(f"  run_id     : {AEP_RECORD['run_id']}")
print(f"  model_id   : {AEP_RECORD['model_id']}")
print(f"  actions    : {len(AEP_RECORD['actions'])}")
print(f"  verifiers  : {len(AEP_RECORD['verifier_results'])}")

# ── Step 2: AEP Validation ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Validate AEP record")
print("=" * 60)

from evomerge.validate.aep import validate_aep_record
result = validate_aep_record(AEP_RECORD)
print(f"  valid_schema          : {result.valid_schema}")
print(f"  has_model_id          : {result.has_model_id}")
print(f"  evidence_completeness : {result.evidence_completeness:.0%}")
print(f"  passed                : {result.passed}")
assert result.passed, f"AEP validation failed: {result.errors}"

# ── Step 3: AgentTrustScore ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Compute AgentTrustScore")
print("=" * 60)

from evomerge.trust_score import AgentTrustScoreBuilder
builder = AgentTrustScoreBuilder()
builder.add_aep_record(AEP_RECORD)
builder.add_task_success(True)
builder.add_benchmark_trust(0.85)
builder.add_receipt(has_receipt=True, digest_verified=True)
trust = builder.build()

print(f"  overall score : {trust.overall:.3f}  (grade {trust.grade})")
print("  breakdown:")
for dim, score in trust.breakdown.items():
    bar = "█" * int(score * 20)
    print(f"    {dim:<28} {score:.2f}  {bar}")

# ── Step 4: Exploit Surface Classification ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Classify benchmark exploit surfaces")
print("=" * 60)

from eval_trust.exploit_surface import classify_findings, EXPLOIT_TAXONOMY

# Simulate a clean task (S6 only — no task manifest)
class _MockFinding:
    def __init__(self, check_id, severity):
        self.check_id = check_id
        self.severity = severity

mock_findings = [_MockFinding("no-task-manifest", "info")]
surfaces = classify_findings(mock_findings)
print(f"  {len(surfaces)} exploit surface(s) detected:")
for s in surfaces:
    print(f"    [{s.severity.upper():8}] {s.surface_id} {s.name}: {s.mitigation[:60]}...")
print(f"  {len(EXPLOIT_TAXONOMY) - len(surfaces)} surface(s) clean (not detected)")

# ── Step 5: Audit Report ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Generate benchmark audit report")
print("=" * 60)

from evomerge.audit_report import AuditReportConfig, generate_audit_report

with tempfile.TemporaryDirectory() as tmp:
    aep_path = Path(tmp) / "aep.jsonl"
    aep_path.write_text(json.dumps(AEP_RECORD) + "\n")

    config = AuditReportConfig(
        title="WasmAgent bscode Audit — recipe17 demo",
        aep_files=[str(aep_path)],
        date="2026-06-25",
    )
    report_md = generate_audit_report(config)
    report_lines = report_md.strip().splitlines()
    print(f"  Generated {len(report_lines)}-line Markdown audit report")
    print(f"  Title: {report_lines[0]}")
    # Show first few lines
    for line in report_lines[:8]:
        print(f"    {line}")

# ── Step 6: Run Receipt ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Build RunReceipt (SCITT-style provenance)")
print("=" * 60)

from evomerge.provenance import RunReceiptBuilder
receipt_builder = RunReceiptBuilder(
    run_id="recipe17-demo-001",
    operator="demo-script",
    notes="P2-4 reproducible demo",
)
receipt_builder.add_model("qwen2.5-1.5b-instruct")
receipt = receipt_builder.build()
print(f"  run_id          : {receipt.run_id}")
print(f"  operator        : {receipt.operator}")
print(f"  evomerge version: {receipt.evomerge_version}")
print(f"  receipt digest  : {receipt.receipt_digest[:24]}...")

# ── Step 7: Evidence Registry ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: Register artifacts in Evidence Registry")
print("=" * 60)

from evomerge.registry import Registry, RegistryEntry

with tempfile.TemporaryDirectory() as tmp:
    reg = Registry(Path(tmp) / "registry")

    reg.register(RegistryEntry(
        id="aep-schema-v0.1",
        entry_type="aep_schema",
        version="0.1.0",
        metadata={"description": "AEP record schema", "schema_version": "aep/v0.1"},
    ))
    reg.register(RegistryEntry(
        id="verifier-build-passes",
        entry_type="verifier",
        version="1.0.0",
        metadata={"description": "Build success verifier for bscode tasks"},
    ))
    reg.register(RegistryEntry(
        id=f"receipt-{receipt.run_id}",
        entry_type="receipt",
        version="0.1.0",
        metadata=receipt.to_dict(),
    ))
    reg.save()

    entries = reg.all()
    print(f"  Registered {len(entries)} entries:")
    for e in entries:
        print(f"    [{e.entry_type:<16}] {e.id} v{e.version}")

    verify_results = reg.verify_entries()
    ok_count = sum(1 for _, ok, _ in verify_results if ok)
    print(f"  Verification: {ok_count}/{len(verify_results)} entries OK")

# ── Step 8: Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8: Summary")
print("=" * 60)
print(f"  AEP validation   : {'PASS' if result.passed else 'FAIL'}")
print(f"  Trust score      : {trust.overall:.3f} (grade {trust.grade})")
print(f"  Exploit surfaces : {len(surfaces)} detected / {len(EXPLOIT_TAXONOMY)} total")
print(f"  Audit report     : {len(report_lines)} lines generated")
print(f"  Run receipt      : {receipt.receipt_digest[:16]}...")
print(f"  Registry entries : {len(entries)} registered")
print("\nAll steps completed successfully.")
