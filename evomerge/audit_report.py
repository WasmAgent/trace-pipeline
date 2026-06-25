"""Benchmark audit report generator.

Combines AEP validation, benchmark linting, and run provenance into a
single structured Markdown report for publication or CI gating.

Usage:
    from evomerge.audit_report import AuditReportConfig, generate_audit_report
    config = AuditReportConfig(title="WasmAgent Audit", aep_files=["data/aep.jsonl"])
    Path("AUDIT_REPORT.md").write_text(generate_audit_report(config))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditReportConfig:
    title: str
    aep_files: list = field(default_factory=list)
    task_dirs: list = field(default_factory=list)
    receipt_paths: list = field(default_factory=list)
    date: str = ""


def _aep_section(aep_files: list) -> str:
    from evomerge.validate.aep import validate_aep_file
    lines = ["## AEP Record Validation\n"]
    if not aep_files:
        lines.append("_No AEP files provided._\n")
        return "\n".join(lines)

    total_pass = total_fail = 0
    rows = []
    for fpath in aep_files:
        p = Path(fpath)
        if not p.exists():
            rows.append(f"| {fpath} | not found | — |")
            continue
        results = validate_aep_file(p)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_pass += passed
        total_fail += total - passed
        avg_ec = (sum(r.evidence_completeness for r in results) / total) if total else 0.0
        rows.append(f"| {fpath} | {passed}/{total} pass | {avg_ec:.0%} |")

    lines.append(f"\n**Overall: {total_pass} pass / {total_fail} fail**\n")
    lines.append("\n| File | Pass rate | Evidence completeness |\n|---|---|---|")
    lines.extend(rows)
    return "\n".join(lines) + "\n"


def _lint_section(task_dirs: list) -> str:
    from evomerge.security.benchmark_linter import lint_benchmark_dir
    lines = ["## Benchmark Trust Scores\n"]
    if not task_dirs:
        lines.append("_No task directories provided._\n")
        return "\n".join(lines)

    lines.append("| Task directory | Trust score | Status | Critical | High |\n|---|---|---|---|---|")
    for tdir in task_dirs:
        result = lint_benchmark_dir(Path(tdir))
        status = "TRUSTED" if result.trusted else "UNTRUSTED"
        lines.append(
            f"| `{tdir}` | {result.score:.2f} | {status} | "
            f"{result.critical_count} | {result.high_count} |"
        )
        for f in result.findings:
            if f.severity in ("critical", "high"):
                lines.append(f"  - [{f.severity.upper()}] {f.check_id}: {f.description}")
    return "\n".join(lines) + "\n"


def _provenance_section(receipt_paths: list) -> str:
    from evomerge.provenance import RunReceipt
    lines = ["## Run Provenance\n"]
    if not receipt_paths:
        lines.append("_No run receipts provided._\n")
        return "\n".join(lines)

    for rpath in receipt_paths:
        p = Path(rpath)
        if not p.exists():
            lines.append(f"- **{rpath}**: not found\n")
            continue
        r = RunReceipt.load(p)
        lines.append(f"### {r.run_id}\n")
        lines.append(f"- Timestamp: {r.timestamp_utc}")
        lines.append(f"- Operator: {r.operator}")
        lines.append(f"- Commit: `{r.repo_commit or 'unknown'}`")
        lines.append(f"- Models: {', '.join(r.model_ids) or 'none'}")
        lines.append(f"- Receipt digest: `{r.receipt_digest[:16]}...`")
        if r.inputs:
            lines.append(f"- Inputs: {len(r.inputs)} file(s)")
        if r.outputs:
            lines.append(f"- Outputs: {len(r.outputs)} file(s)\n")
    return "\n".join(lines) + "\n"


def _standards_section(aep_files: list) -> str:
    """Generate OWASP/MCP/OTel standards coverage matrix from AEP records."""
    lines = ["## Standards Coverage Matrix\n"]
    lines.append("Checks AEP evidence against OWASP MCP Top 10 and WasmAgent control coverage.\n")

    # Control coverage requirements
    CONTROLS = [
        ("OWASP-MCP-01", "Tool Poisoning", "tool_manifest_digest"),
        ("OWASP-MCP-02", "Scope Creep", "capability_decisions"),
        ("OWASP-MCP-03", "Rug Pull", "tool_manifest_digest"),
        ("OWASP-MCP-05", "Taint Passthrough", "input_refs[*].taint_labels"),
        ("OWASP-MCP-07", "Supply Chain", "repo_commit"),
        ("OTel-GenAI", "Observability Export", "trace_id"),
        ("AEP-Provenance", "Action Provenance", "actions[*].result_digest"),
    ]

    if not aep_files:
        lines.append("_No AEP files provided — cannot assess standards coverage._\n")
        return "\n".join(lines)

    # Aggregate field presence across all records
    field_presence: dict[str, int] = {}
    total_records = 0
    for fpath in aep_files:
        p = Path(fpath)
        if not p.exists():
            continue
        import json
        try:
            with open(p) as f:
                records = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
        total_records += len(records)
        for rec in records:
            if rec.get("tool_manifest_digest"):
                field_presence["tool_manifest_digest"] = field_presence.get("tool_manifest_digest", 0) + 1
            if rec.get("capability_decisions"):
                field_presence["capability_decisions"] = field_presence.get("capability_decisions", 0) + 1
            if rec.get("repo_commit"):
                field_presence["repo_commit"] = field_presence.get("repo_commit", 0) + 1
            if rec.get("trace_id"):
                field_presence["trace_id"] = field_presence.get("trace_id", 0) + 1
            if rec.get("input_refs"):
                tainted = any(r.get("taint_labels") for r in rec.get("input_refs", []))
                if tainted:
                    field_presence["input_refs[*].taint_labels"] = field_presence.get("input_refs[*].taint_labels", 0) + 1
            if any(a.get("result_digest") for a in rec.get("actions", [])):
                field_presence["actions[*].result_digest"] = field_presence.get("actions[*].result_digest", 0) + 1

    lines.append(f"\n**Total records analysed: {total_records}**\n")
    lines.append("\n| Control | Risk | Evidence Field | Coverage | Status |\n|---|---|---|---|---|")
    for ctrl_id, risk, evidence_field in CONTROLS:
        count = field_presence.get(evidence_field, 0)
        pct = (count / total_records * 100) if total_records > 0 else 0.0
        status = "✓ Covered" if pct >= 50 else ("△ Partial" if pct > 0 else "✗ Gap")
        lines.append(f"| {ctrl_id} | {risk} | `{evidence_field}` | {count}/{total_records} ({pct:.0f}%) | {status} |")

    return "\n".join(lines) + "\n"


def generate_audit_report(config: AuditReportConfig) -> str:
    import time
    date = config.date or time.strftime("%Y-%m-%d")
    sections = [
        f"# {config.title}\n\n> Generated {date}\n",
        _aep_section(config.aep_files),
        _standards_section(config.aep_files),
        _lint_section(config.task_dirs),
        _provenance_section(config.receipt_paths),
        "---\n_Generated by evomerge audit-report_\n",
    ]
    return "\n".join(sections)
