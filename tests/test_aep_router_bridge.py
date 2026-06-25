"""Tests for AEP → router feature bridge."""
from __future__ import annotations

import pytest

from evomerge.router.aep_bridge import features_from_aep, label_from_aep
from evomerge.router.labels import RouterLabel


# ---------------------------------------------------------------------------
# label_from_aep tests
# ---------------------------------------------------------------------------

def test_clean_aep_small_model_can_handle():
    """All verifiers pass, no deny decisions → small_model_can_handle."""
    record = {
        "verifier_results": [
            {"verifier_id": "v1", "passed": True},
            {"verifier_id": "v2", "passed": True},
        ],
        "capability_decisions": [
            {"capability": "fs:read", "decision": "allow"},
        ],
    }
    assert label_from_aep(record) == RouterLabel.small_model_can_handle


def test_failed_verifier_need_repair():
    """One failed verifier, no deny decisions → need_repair."""
    record = {
        "verifier_results": [
            {"verifier_id": "v1", "passed": True},
            {"verifier_id": "v2", "passed": False},
        ],
        "capability_decisions": [],
    }
    assert label_from_aep(record) == RouterLabel.need_repair


def test_deny_decision_need_large_model():
    """Exactly one deny decision → need_large_model."""
    record = {
        "verifier_results": [],
        "capability_decisions": [
            {"capability": "fs:write", "decision": "deny"},
        ],
    }
    assert label_from_aep(record) == RouterLabel.need_large_model


def test_multiple_denies_need_human_review():
    """Two or more deny decisions → need_human_review."""
    record = {
        "verifier_results": [],
        "capability_decisions": [
            {"capability": "fs:write", "decision": "deny"},
            {"capability": "net:connect", "decision": "deny"},
        ],
    }
    assert label_from_aep(record) == RouterLabel.need_human_review


# ---------------------------------------------------------------------------
# features_from_aep tests
# ---------------------------------------------------------------------------

def test_features_from_aep_tool_counts():
    """actions with state_changing=True and no error → counts flow correctly."""
    record = {
        "actions": [
            {"tool_name": "write_file", "state_changing": True},
            {"tool_name": "read_file", "state_changing": False},
            {"tool_name": "delete_file", "state_changing": True, "error": "not found"},
        ],
        "capability_decisions": [],
        "verifier_results": [],
    }
    features = features_from_aep(record)
    assert features.eval_tool_calls_total == 3
    # state_changing=True + no error: only write_file qualifies
    assert features.eval_tool_calls_valid == 1
    assert features.eval_tool_validity_rate == pytest.approx(1 / 3)
    assert features.taskspec_has_tools == 1
    assert features.eval_escalated == 0


def test_features_from_aep_empty():
    """Empty AEP record → all zero counts, tool_validity_rate defaults to 1.0."""
    record: dict = {}
    features = features_from_aep(record)
    assert features.eval_tool_calls_total == 0
    assert features.eval_tool_calls_valid == 0
    assert features.eval_tool_validity_rate == 1.0
    assert features.taskspec_has_tools == 0
    assert features.eval_violation_count == 0
    assert features.eval_hard_violation_count == 0
    assert features.eval_escalated == 0
    # Constraint fields always 0 for AEP records
    assert features.taskspec_n_constraints == 0
    assert features.taskspec_n_hard == 0
