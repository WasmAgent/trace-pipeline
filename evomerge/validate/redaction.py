"""evomerge.validate.redaction — redaction report schema and helper.

A RedactionReport describes what PII patterns were checked during export,
how many records were scanned, and which fields were redacted.  It is written
alongside contamination_report.json and schema_report.json so downstream
consumers can audit the trust level of each exported batch.

The report is purely descriptive — it records *what the exporter did*, not
whether any actual PII was found (that would require inspecting content we are
trying not to log).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RedactionReport:
    """Describes the redaction pass applied to a training-data export batch.

    Attributes:
        redaction_version:  opaque version string identifying the redaction
                            logic (e.g. "bscode/pii-redact/v1").
        evidence_source:    where the data came from, e.g. "client_reported"
                            or "replay_verified".
        fields_redacted:    list of field names that were scanned and
                            potentially redacted (order is informational).
        patterns_applied:   human-readable names of the PII patterns that
                            were searched (e.g. "JWT", "API_KEY", "EMAIL").
        n_records_scanned:  total number of records inspected.
        n_fields_modified:  number of (record, field) pairs where at least
                            one replacement was made.  0 means no PII was
                            detected, not that redaction was skipped.
    """

    redaction_version: str
    evidence_source: str
    fields_redacted: list[str] = field(default_factory=list)
    patterns_applied: list[str] = field(default_factory=list)
    n_records_scanned: int = 0
    n_fields_modified: int = 0

    def to_dict(self) -> dict:
        return {
            "redaction_version": self.redaction_version,
            "evidence_source": self.evidence_source,
            "fields_redacted": self.fields_redacted,
            "patterns_applied": self.patterns_applied,
            "n_records_scanned": self.n_records_scanned,
            "n_fields_modified": self.n_fields_modified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RedactionReport":
        return cls(
            redaction_version=d.get("redaction_version", "unknown"),
            evidence_source=d.get("evidence_source", "unknown"),
            fields_redacted=d.get("fields_redacted", []),
            patterns_applied=d.get("patterns_applied", []),
            n_records_scanned=d.get("n_records_scanned", 0),
            n_fields_modified=d.get("n_fields_modified", 0),
        )


# Standard field list for bscode rollout-wire/v1 exports
BSCODE_REDACTED_FIELDS = [
    "task",
    "final_answer",
    "tool_call_sequence",
    "build_result.stderr",
]

# Standard pattern names matching bscode/pii-redact/v1 PII_PATTERNS
BSCODE_PATTERNS = ["JWT", "API_KEY", "EMAIL"]


__all__ = [
    "RedactionReport",
    "BSCODE_REDACTED_FIELDS",
    "BSCODE_PATTERNS",
]
