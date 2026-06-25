"""Tests for benchmark audit report generator."""
import json
import tempfile
from pathlib import Path

from evomerge.audit_report import AuditReportConfig, generate_audit_report


def test_empty_config():
    config = AuditReportConfig(title="Test Audit", date="2026-06-25")
    report = generate_audit_report(config)
    assert "Test Audit" in report
    assert "2026-06-25" in report
    assert "No AEP files provided" in report
    assert "No task directories provided" in report


def test_with_aep_file():
    with tempfile.TemporaryDirectory() as tmp:
        aep_path = Path(tmp) / "aep.jsonl"
        aep_path.write_text(
            json.dumps({
                "schema_version": "aep/v0.1",
                "run_id": "test-001",
                "created_at_ms": 1000,
                "model_id": "qwen",
            }) + "\n"
        )
        config = AuditReportConfig(title="T", aep_files=[str(aep_path)], date="2026-06-25")
        report = generate_audit_report(config)
        assert "AEP Record Validation" in report
        assert "1 pass" in report


def test_with_task_dir():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "task.json").write_text('{"name":"test"}')
        config = AuditReportConfig(title="T", task_dirs=[tmp], date="2026-06-25")
        report = generate_audit_report(config)
        assert "Benchmark Trust Scores" in report
        assert "TRUSTED" in report


def test_with_receipt():
    from evomerge.provenance import RunReceiptBuilder
    with tempfile.TemporaryDirectory() as tmp:
        receipt_path = Path(tmp) / "receipt.json"
        RunReceiptBuilder(run_id="audit-001", operator="ci").build().save(receipt_path)
        config = AuditReportConfig(title="T", receipt_paths=[str(receipt_path)], date="2026-06-25")
        report = generate_audit_report(config)
        assert "Run Provenance" in report
        assert "audit-001" in report
