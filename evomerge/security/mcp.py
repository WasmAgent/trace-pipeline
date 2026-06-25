"""MCP security evaluation record — schema for logging firewall decisions.

Records one security evaluation event per tool call attempt.
Consumed by evomerge pipeline to build MCP security training datasets
and benchmark against MCPTox/AgentDojo/MCP-SafetyBench style tasks.

Mirrors the TypeScript types in @wasmagent/mcp-firewall but uses
Pydantic for Python-side validation.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RiskCategory(str, Enum):
    tool_poisoning = "tool_poisoning"
    shadowing = "shadowing"
    rug_pull = "rug_pull"
    exfiltration = "exfiltration"
    sampling_abuse = "sampling_abuse"
    invisible_chars = "invisible_chars"


class FirewallDecision(str, Enum):
    allow = "allow"
    deny = "deny"
    ask_user = "ask_user"
    dry_run = "dry_run"


class ToolRiskFinding(BaseModel):
    severity: RiskSeverity
    category: RiskCategory
    field: str                   # "name" | "description" | "inputSchema"
    evidence_excerpt: str
    evidence_hash: str
    recommendation: str          # "allow" | "ask" | "deny"


class McpSecurityEvalRecord(BaseModel):
    """One security evaluation event for an MCP tool call attempt.

    schema_version is always "mcp-security-eval/v1".
    """

    schema_version: str = "mcp-security-eval/v1"
    record_id: str
    session_id: str
    tool_name: str
    server_id: str

    # Descriptor integrity
    descriptor_snapshot_hash: str
    rug_pull_detected: bool = False
    rug_pull_field: str | None = None   # "description" | "inputSchema" | None

    # Static vetting
    risk_findings: list[ToolRiskFinding] = Field(default_factory=list)
    vetting_recommendation: str = "allow"   # "allow" | "ask" | "deny"

    # Per-call policy
    firewall_decision: FirewallDecision = FirewallDecision.allow
    policy_ids_matched: list[str] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)

    # Consent
    consent_ref: str | None = None
    consent_required: bool = False

    # Outcome (filled after call completes or is blocked)
    call_was_blocked: bool = False
    taint_instruction_like: bool = False

    # Metadata
    evaluated_at_ms: int           # Unix ms
    task_id: str | None = None
    rollout_id: str | None = None


__all__ = [
    "FirewallDecision",
    "McpSecurityEvalRecord",
    "RiskCategory",
    "RiskSeverity",
    "ToolRiskFinding",
]
