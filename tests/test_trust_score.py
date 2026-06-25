"""Tests for AgentTrustScore."""
import math
from evomerge.trust_score import (
    AgentTrustScore, AgentTrustScoreBuilder, compute_trust_score, _geometric_mean
)


def test_geometric_mean_empty():
    assert _geometric_mean([]) == 1.0


def test_geometric_mean_zeros():
    assert _geometric_mean([0.5, 0.0, 0.8]) == 0.0


def test_geometric_mean_values():
    result = _geometric_mean([1.0, 1.0, 1.0])
    assert abs(result - 1.0) < 1e-9


def test_grade_mapping():
    assert AgentTrustScore(0.95, {}).grade == "A"
    assert AgentTrustScore(0.80, {}).grade == "B"
    assert AgentTrustScore(0.65, {}).grade == "C"
    assert AgentTrustScore(0.45, {}).grade == "D"
    assert AgentTrustScore(0.30, {}).grade == "F"


def test_clean_aep_scores_1():
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "t1",
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": [{"verifier_id": "v1", "passed": True}],
    }
    score = compute_trust_score(aep_record=record, task_passed=True, has_receipt=True)
    assert score.breakdown["task_success"] == 1.0
    assert score.breakdown["verifier_agreement"] == 1.0
    assert score.breakdown["policy_compliance"] == 1.0
    assert score.overall > 0.9


def test_deny_decision_lowers_score():
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "t2",
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [
            {"capability": "tool:bash", "subject": "a", "resource": "bash", "decision": "deny"}
        ],
        "verifier_results": [],
    }
    score = compute_trust_score(aep_record=record, task_passed=False)
    assert score.breakdown["policy_compliance"] == 0.0
    assert score.overall == 0.0


def test_failed_verifier_lowers_score():
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "t3",
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": [
            {"verifier_id": "v1", "passed": True},
            {"verifier_id": "v2", "passed": False},
        ],
    }
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(record)
    score = builder.build()
    assert score.breakdown["verifier_agreement"] == 0.5


def test_budget_compliance():
    record = {
        "schema_version": "aep/v0.1",
        "run_id": "t4",
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": [],
        "budget_ledger": {
            "token_budget": {"limit": 1000, "spent": 1500},
        },
    }
    builder = AgentTrustScoreBuilder()
    builder.add_aep_record(record)
    score = builder.build()
    assert score.breakdown["budget_compliance"] == 0.0


def test_to_dict_has_grade():
    score = compute_trust_score(task_passed=True, has_receipt=True)
    d = score.to_dict()
    assert "grade" in d
    assert "overall" in d
    assert "breakdown" in d


def test_add_dimension_custom():
    builder = AgentTrustScoreBuilder()
    builder.add_dimension("custom_check", 0.8)
    score = builder.build()
    assert score.breakdown["custom_check"] == 0.8
