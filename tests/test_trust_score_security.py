"""Security-focused tests for AgentTrustScore.

Verifies that the trust score system cannot be inflated by:
  (a) Empty / fully absent AEP evidence
  (b) Missing capability decisions
  (c) Tampered receipt content
  (d) Mixing one benign record with one empty/malicious record
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from evomerge.trust_score import (
    AgentTrustScoreBuilder,
    _geometric_mean,
)


# ---------------------------------------------------------------------------
# (a) Empty record → overall must NOT be 1.0
# ---------------------------------------------------------------------------

def test_empty_aep_record_does_not_return_1_0():
    """An AEP record with all lists empty must not produce overall == 1.0.

    Before the fix, missing evidence dimensions defaulted to 1.0 which
    caused geometric_mean([1.0, 1.0, 1.0, ...]) == 1.0 for blank records.
    After the fix, missing dims are None and excluded from the mean.
    """
    empty_record: dict = {
        "schema_version": "aep/v0.1",
        "run_id": "empty-run",
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": [],
    }
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(empty_record)
    score = builder.build()

    # All three AEP-derived dimensions should be unknown (None)
    assert score.breakdown.get("evidence_completeness") is None, (
        "No state-changing actions → evidence_completeness must be None"
    )
    assert score.breakdown.get("policy_compliance") is None, (
        "No capability_decisions → policy_compliance must be None"
    )
    assert score.breakdown.get("verifier_agreement") is None, (
        "No verifier_results → verifier_agreement must be None"
    )

    # With no known dimensions overall should also be None (or if supply_chain
    # was added, must not be 1.0)
    if score.overall is not None:
        assert score.overall != 1.0, (
            "Overall score for an empty record must not be 1.0"
        )


def test_totally_empty_builder_returns_none_overall():
    """A builder with zero dimensions added must return overall=None."""
    score = AgentTrustScoreBuilder().build()
    assert score.overall is None, (
        "Builder with no dimensions must return overall=None, not 1.0"
    )


def test_geometric_mean_empty_raises():
    """_geometric_mean([]) must raise ValueError, not silently return 1.0."""
    with pytest.raises(ValueError, match="empty list"):
        _geometric_mean([])


# ---------------------------------------------------------------------------
# (b) No capability decisions → policy_compliance must NOT be 1.0
# ---------------------------------------------------------------------------

def test_no_capability_decisions_policy_compliance_unknown():
    """When capability_decisions list is absent/empty, policy_compliance is None.

    An agent that never recorded any capability gate cannot claim a perfect
    policy compliance score — we simply don't know.
    """
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "no-caps",
        "created_at_ms": 0,
        "actions": [
            {"action_id": "a1", "tool": "bash", "state_changing": True,
             "result_digest": "abc123"}
        ],
        "capability_decisions": [],  # empty — no decisions recorded
        "verifier_results": [{"verifier_id": "v1", "passed": True}],
    }
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(record)
    score = builder.build()

    assert score.breakdown.get("policy_compliance") is None, (
        "Empty capability_decisions must yield policy_compliance=None, not 1.0"
    )
    assert score.breakdown.get("policy_compliance") != 1.0


def test_no_capability_decisions_field_missing():
    """capability_decisions key entirely missing → policy_compliance is None."""
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "missing-caps",
        "created_at_ms": 0,
        "actions": [],
        # capability_decisions key not present at all
        "verifier_results": [],
    }
    score = AgentTrustScoreBuilder().add_aep_record(record).build()
    assert score.breakdown.get("policy_compliance") is None


# ---------------------------------------------------------------------------
# (c) Tampered receipt → digest_verified must be False
# ---------------------------------------------------------------------------

def _make_receipt_file(tamper: bool = False) -> Path:
    """Create a temporary receipt JSON file.

    The RunReceipt dataclass does not yet self-embed receipt_digest in its
    to_dict() output, so we craft a minimal receipt JSON manually that
    includes the field for verification testing.
    """
    body = {
        "receipt_version": "run-receipt/v0.1",
        "run_id": "test-run-001",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "operator": "ci",
        "repo_commit": "abc123",
        "evomerge_version": "0.1.0",
        "inputs": [],
        "outputs": [],
        "model_ids": [],
        "policy_bundle_digest": "",
        "notes": "",
    }
    # Compute the canonical digest of the body (no whitespace, sorted keys)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    correct_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if tamper:
        # Write correct body but with a wrong digest
        full = {**body, "receipt_digest": "0" * 64}
    else:
        full = {**body, "receipt_digest": correct_digest}

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(full, tmp)
    tmp.close()
    return Path(tmp.name)


def test_valid_receipt_digest_verified():
    """A receipt with matching digest → supply_chain_integrity == 1.0."""
    receipt_path = _make_receipt_file(tamper=False)
    try:
        builder = AgentTrustScoreBuilder()
        builder.add_receipt_path(receipt_path)
        score = builder.build()
        assert score.breakdown["supply_chain_integrity"] == 1.0, (
            "Valid receipt with matching digest must give supply_chain_integrity=1.0"
        )
    finally:
        receipt_path.unlink(missing_ok=True)


def test_tampered_receipt_digest_fails():
    """A receipt whose digest field does not match the body must NOT get 1.0."""
    receipt_path = _make_receipt_file(tamper=True)
    try:
        builder = AgentTrustScoreBuilder()
        builder.add_receipt_path(receipt_path)
        score = builder.build()
        integrity = score.breakdown["supply_chain_integrity"]
        assert integrity != 1.0, (
            "Tampered receipt (wrong digest) must not give supply_chain_integrity=1.0"
        )
        assert integrity is not None and integrity < 1.0, (
            f"Expected supply_chain_integrity < 1.0 for tampered receipt, got {integrity}"
        )
    finally:
        receipt_path.unlink(missing_ok=True)


def test_receipt_missing_digest_field_not_verified():
    """A receipt that has no receipt_digest field must not pass digest verification."""
    body = {
        "receipt_version": "run-receipt/v0.1",
        "run_id": "legacy-receipt",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "operator": "ci",
        "repo_commit": "",
        "evomerge_version": "0.0.1",
        "inputs": [],
        "outputs": [],
        "model_ids": [],
        "policy_bundle_digest": "",
        "notes": "",
        # no receipt_digest field
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(body, tmp)
    tmp.close()
    receipt_path = Path(tmp.name)
    try:
        builder = AgentTrustScoreBuilder()
        builder.add_receipt_path(receipt_path)
        score = builder.build()
        # Legacy receipt without digest field → should be 0.7 (unverified), not 1.0
        assert score.breakdown["supply_chain_integrity"] != 1.0
    finally:
        receipt_path.unlink(missing_ok=True)


def test_nonexistent_receipt_path():
    """Pointing add_receipt_path() at a non-existent file must not give 1.0."""
    builder = AgentTrustScoreBuilder()
    builder.add_receipt_path(Path("/tmp/__nonexistent_receipt_xyz__.json"))
    score = builder.build()
    assert score.breakdown["supply_chain_integrity"] != 1.0
    assert score.breakdown["supply_chain_integrity"] == 0.5


# ---------------------------------------------------------------------------
# (d) One benign + one malicious (fully empty) record cannot pull score to A
# ---------------------------------------------------------------------------

_BENIGN_RECORD = {
    "schema_version": "aep/v0.1",
    "run_id": "benign-run",
    "created_at_ms": 0,
    "actions": [
        {"action_id": "a1", "tool": "bash", "state_changing": True,
         "result_digest": "deadbeef"},
        {"action_id": "a2", "tool": "read_file", "state_changing": False},
    ],
    "capability_decisions": [
        {"capability": "tool:bash", "subject": "agent", "resource": "bash",
         "decision": "allow"},
    ],
    "verifier_results": [
        {"verifier_id": "v1", "passed": True},
        {"verifier_id": "v2", "passed": True},
    ],
}

_EMPTY_RECORD = {
    "schema_version": "aep/v0.1",
    "run_id": "malicious-empty-run",
    "created_at_ms": 0,
    "actions": [],
    "capability_decisions": [],
    "verifier_results": [],
}


def test_mixed_benign_and_empty_record_no_A_grade():
    """A mix of one benign and one fully empty record must not produce grade A.

    Grade A requires overall >= 0.9 AND >= 6 known dimensions.  An empty
    record contributes None dimensions which dilute evidence; the overall
    score cannot legitimately reach A in such a mix.
    """
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(_BENIGN_RECORD)
    builder.add_aep_record(_EMPTY_RECORD)
    score = builder.build()

    assert score.grade != "A", (
        f"Mixed benign+empty records must not reach grade A, got {score.grade} "
        f"(overall={score.overall}, breakdown={score.breakdown})"
    )


def test_mixed_benign_and_empty_policy_compliance_not_1():
    """After adding a benign then empty record, policy_compliance must not be 1.0.

    The empty record has no capability_decisions, so the second call to
    add_aep_record() overwrites policy_compliance to None (unknown).
    """
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(_BENIGN_RECORD)
    builder.add_aep_record(_EMPTY_RECORD)
    score = builder.build()

    # After the empty record is processed, policy_compliance should be None
    assert score.breakdown.get("policy_compliance") is None, (
        "After empty record overwrites benign, policy_compliance must be None"
    )
