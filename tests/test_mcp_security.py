"""Tests for evomerge.security.mcp — McpSecurityEvalRecord schema."""
from __future__ import annotations


from evomerge.security.mcp import (
    FirewallDecision,
    McpSecurityEvalRecord,
    RiskCategory,
    RiskSeverity,
    ToolRiskFinding,
)


def _minimal_record(**kwargs) -> McpSecurityEvalRecord:
    base = dict(
        record_id="rec-001",
        session_id="sess-001",
        tool_name="run_code",
        server_id="srv-001",
        descriptor_snapshot_hash="abc123",
        evaluated_at_ms=1_700_000_000_000,
    )
    base.update(kwargs)
    return McpSecurityEvalRecord(**base)


def test_minimal_record_defaults():
    r = _minimal_record()
    assert r.schema_version == "mcp-security-eval/v1"
    assert r.rug_pull_detected is False
    assert r.firewall_decision == FirewallDecision.allow
    assert r.call_was_blocked is False
    assert r.risk_findings == []


def test_blocked_record():
    r = _minimal_record(
        call_was_blocked=True,
        firewall_decision=FirewallDecision.deny,
        risk_findings=[
            ToolRiskFinding(
                severity=RiskSeverity.critical,
                category=RiskCategory.tool_poisoning,
                field="description",
                evidence_excerpt="ignore previous instructions",
                evidence_hash="dead1234",
                recommendation="deny",
            )
        ],
    )
    assert r.call_was_blocked is True
    assert r.risk_findings[0].category == RiskCategory.tool_poisoning


def test_rug_pull_fields():
    r = _minimal_record(
        rug_pull_detected=True,
        rug_pull_field="description",
        firewall_decision=FirewallDecision.ask_user,
    )
    assert r.rug_pull_detected is True
    assert r.rug_pull_field == "description"


def test_serialization_round_trip():
    r = _minimal_record(
        consent_ref="snap-abc",
        consent_required=True,
        policy_ids_matched=["deny-blocked-vetting"],
        rollout_id="rollout-001",
    )
    data = r.model_dump()
    r2 = McpSecurityEvalRecord(**data)
    assert r2.consent_ref == "snap-abc"
    assert r2.policy_ids_matched == ["deny-blocked-vetting"]


def test_firewall_decision_enum_values():
    assert set(FirewallDecision) == {
        FirewallDecision.allow,
        FirewallDecision.deny,
        FirewallDecision.ask_user,
        FirewallDecision.dry_run,
    }


def test_risk_category_enum_values():
    categories = {c.value for c in RiskCategory}
    assert "tool_poisoning" in categories
    assert "rug_pull" in categories
    assert "exfiltration" in categories
