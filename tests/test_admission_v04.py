"""Tests for AEP v0.3/v0.4 admission through the quality gate."""
from __future__ import annotations

from evomerge.validate.quality_gate import compute_admission_score


def _make_aep_record(schema_version="aep/v0.4", recording_mode=None):
    """Create a minimal valid AEP record for testing admission."""
    record = {
        "schema_version": schema_version,
        "run_id": "run-test-001",
        "created_at_ms": 1700000000000,
        "repo_commit": "abc123def456",
        "model_id": "gpt-4-turbo",
        "tool_manifest_digest": "sha256:deadbeef",
        "actions": [
            {
                "action_id": "act-1",
                "tool_name": "file_write",
                "state_changing": True,
                "timestamp_ms": 1700000001000,
                "result_digest": "sha256:cafebabe",
                "precondition_digest": "sha256:feed0001",
            }
        ],
        "capability_decisions": [
            {
                "capability": "file_write",
                "subject": "agent-1",
                "resource": "/tmp/out.txt",
                "decision": "allow",
            }
        ],
        "verifier_results": [
            {"verifier_id": "determinism-v1", "passed": True, "score": 0.95}
        ],
    }
    if recording_mode is not None:
        record["recording_mode"] = recording_mode
    return record


class TestAepV04Admission:
    """Verify that v0.3 and v0.4 records are not rejected by the admission gate."""

    def test_v04_record_not_rejected(self):
        record = _make_aep_record(schema_version="aep/v0.4")
        result = compute_admission_score(record)
        assert result["score"] > 0.0, "v0.4 record should not be rejected with score 0.0"
        assert result["category"] != "reject"

    def test_v03_record_not_rejected(self):
        record = _make_aep_record(schema_version="aep/v0.3")
        result = compute_admission_score(record)
        assert result["score"] > 0.0, "v0.3 record should not be rejected with score 0.0"
        assert result["category"] != "reject"

    def test_v01_still_accepted(self):
        record = _make_aep_record(schema_version="aep/v0.1")
        result = compute_admission_score(record)
        assert result["score"] > 0.0

    def test_v02_still_accepted(self):
        record = _make_aep_record(schema_version="aep/v0.2")
        result = compute_admission_score(record)
        assert result["score"] > 0.0

    def test_invalid_version_rejected(self):
        record = _make_aep_record(schema_version="aep/v9.9")
        result = compute_admission_score(record)
        assert result["score"] == 0.0
        assert result["category"] == "reject"

    def test_v04_with_dsse_envelope(self):
        record = _make_aep_record(schema_version="aep/v0.4")
        record["dsse_envelope"] = {
            "payloadType": "application/vnd.aep.record+json",
            "payload": "base64data",
            "signatures": [{"keyid": "key-1", "sig": "sigdata"}],
        }
        result = compute_admission_score(record)
        assert result["score"] > 0.0
        assert result["category"] != "reject"


class TestRecordingModeBonus:
    """Verify that recording_mode adjusts the admission score."""

    def test_full_mode_gives_bonus(self):
        base = _make_aep_record(schema_version="aep/v0.3", recording_mode=None)
        full = _make_aep_record(schema_version="aep/v0.3", recording_mode="full")
        base_result = compute_admission_score(base)
        full_result = compute_admission_score(full)
        assert full_result["score"] > base_result["score"]
        assert full_result["score"] - base_result["score"] == pytest.approx(0.05, abs=1e-4)

    def test_delta_mode_gives_bonus(self):
        base = _make_aep_record(schema_version="aep/v0.3", recording_mode=None)
        delta = _make_aep_record(schema_version="aep/v0.3", recording_mode="delta")
        base_result = compute_admission_score(base)
        delta_result = compute_admission_score(delta)
        assert delta_result["score"] > base_result["score"]
        assert delta_result["score"] - base_result["score"] == pytest.approx(0.02, abs=1e-4)

    def test_validation_mode_no_bonus(self):
        base = _make_aep_record(schema_version="aep/v0.3", recording_mode=None)
        val = _make_aep_record(schema_version="aep/v0.3", recording_mode="validation")
        base_result = compute_admission_score(base)
        val_result = compute_admission_score(val)
        assert val_result["score"] == base_result["score"]

    def test_absent_mode_no_bonus(self):
        record = _make_aep_record(schema_version="aep/v0.3", recording_mode=None)
        result = compute_admission_score(record)
        # Ensure it doesn't crash and score is reasonable
        assert result["score"] > 0.0

    def test_full_bonus_capped_at_1(self):
        """Score should not exceed 1.0 even with bonus."""
        record = _make_aep_record(schema_version="aep/v0.4", recording_mode="full")
        result = compute_admission_score(record)
        assert result["score"] <= 1.0


# Need pytest for approx
import pytest
