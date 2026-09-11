"""Tests for aep/v0.5 attribution grading in the AEP validation result
and the trust-score builder.

Canonical vocabulary: wasmagent-protocol 0.1.9 (aep/v0.5) — authorized_by,
authority_origin, identity_source, attribution_backing,
run_attribution_backing_floor, run_attribution_backing_observed.
"""
from __future__ import annotations

from typing import Any

from evomerge.trust_score import AgentTrustScoreBuilder
from evomerge.validate.aep import validate_aep_record


def _base_record(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "aep/v0.5",
        "run_id": "run-v05-001",
        "created_at_ms": 1_757_460_000_000,
        "actions": [],
    }
    base.update(kwargs)
    return base


def _strong_attribution(**kwargs: Any) -> dict[str, Any]:
    fields = {
        "user_id": "user-dana@acme.example",
        "authorized_by": "manager-ade@acme.example",
        "authority_origin": "subject_consented",
        "identity_source": "qualified_certificate",
        "attribution_backing": "qualified_signature",
        "run_attribution_backing_floor": "qualified_signature",
        "run_attribution_backing_observed": ["qualified_signature"],
    }
    fields.update(kwargs)
    return fields


class TestAttributionExtraction:
    def test_v0_5_fields_extracted_into_attribution_block(self) -> None:
        record = _base_record(**_strong_attribution())
        result = validate_aep_record(record)
        assert result.v0_5_fields_count == 6
        assert result.attribution is not None
        assert result.attribution["authorized_by"] == "manager-ade@acme.example"
        assert result.attribution["authority_origin"] == "subject_consented"

    def test_absent_grading_yields_none(self) -> None:
        result = validate_aep_record(_base_record())
        assert result.attribution is None
        assert result.v0_5_fields_count == 0

    def test_partial_grading_is_carried_verbatim(self) -> None:
        record = _base_record(authority_origin="unknown", attribution_backing="unknown")
        result = validate_aep_record(record)
        assert result.v0_5_fields_count == 2
        assert result.attribution is not None
        assert result.attribution["authority_origin"] == "unknown"


class TestAttributionTrustDimension:
    def test_strong_grading_scores_high(self) -> None:
        record = _base_record(**_strong_attribution())
        score = AgentTrustScoreBuilder().add_aep_record(record).build()
        assert score.breakdown["attribution_integrity"] is not None
        assert score.breakdown["attribution_integrity"] >= 0.9

    def test_unknown_axes_score_below_neutral(self) -> None:
        record = _base_record(
            authority_origin="unknown",
            attribution_backing="unknown",
        )
        score = AgentTrustScoreBuilder().add_aep_record(record).build()
        assert score.breakdown["attribution_integrity"] is not None
        assert score.breakdown["attribution_integrity"] < 0.5

    def test_absent_grading_is_none_with_note(self) -> None:
        record = _base_record()
        score = AgentTrustScoreBuilder().add_aep_record(record).build()
        assert score.breakdown["attribution_integrity"] is None
        assert any("attribution_integrity" in n for n in score.notes)

    def test_admin_assigned_grading_is_partial_not_strong(self) -> None:
        # Admin-granted authority is real but is NOT interactive subject
        # consent — the Entra finding. It must not score as fully verified.
        fields = _strong_attribution(
            authority_origin="administrator_assigned",
            attribution_backing="principal_key_signed",
        )
        record = _base_record(**fields)
        admin = AgentTrustScoreBuilder().add_aep_record(record).build()
        consented = AgentTrustScoreBuilder().add_aep_record(
            _base_record(**_strong_attribution())
        ).build()
        assert admin.breakdown["attribution_integrity"] is not None
        assert consented.breakdown["attribution_integrity"] is not None
        assert admin.breakdown["attribution_integrity"] < consented.breakdown["attribution_integrity"]
