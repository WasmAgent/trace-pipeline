"""Round-trip test: a ComplianceEvalRecord produced against the canonical schema
serialises to JSON and deserialises back to an equivalent object without loss."""

from __future__ import annotations

import json

from evomerge.schemas.compliance import (
    ComplianceError,
    ComplianceEvalRecord,
    ConstraintCategory,
    ConstraintLevel,
    ConstraintViolation,
    EvidenceSpan,
    RepairStrategy,
    RepairTraceEntry,
    RunMode,
    TokenCost,
    ViolationStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_record() -> ComplianceEvalRecord:
    """Minimal valid ComplianceEvalRecord (all required fields set)."""
    return ComplianceEvalRecord(
        task_id="task-001",
        task_spec_hash="abc123",
        model="gpt-4o",
        mode=RunMode.direct,
        final_pass=True,
        artifact="The answer is 42.",
    )


def _full_record() -> ComplianceEvalRecord:
    """ComplianceEvalRecord with all optional fields populated."""
    violation = ConstraintViolation(
        constraint_id="c-format-1",
        level=ConstraintLevel.hard,
        category=ConstraintCategory.format,
        hint="Output must start with 'Answer:'",
        evidence_span=EvidenceSpan(char_range=(0, 6), line_range=(1, 1)),
        detected_at=ViolationStage.post_decode,
    )
    repair = RepairTraceEntry(
        round=1,
        violation_ids=["c-format-1"],
        strategy=RepairStrategy.patch,
        ok=True,
        rolled_back=False,
        remaining_violation_ids=[],
        token_cost={"prompt": 120, "generation": 40},
        latency_ms=210.5,
    )
    return ComplianceEvalRecord(
        task_id="task-002",
        task_spec_hash="def456",
        model="claude-3-5-sonnet",
        mode=RunMode.full_pcl,
        violations=[violation],
        repair_trace=[repair],
        repair_rounds=1,
        final_pass=True,
        token_cost=TokenCost(prompt=800, generation=200, repair=160),
        latency_ms=1342.7,
        artifact="Answer: The answer is 42.",
        error=None,
    )


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

def test_minimal_record_json_round_trip() -> None:
    record = _minimal_record()
    raw = record.model_dump_json()
    restored = ComplianceEvalRecord.model_validate_json(raw)

    assert restored.task_id == record.task_id
    assert restored.task_spec_hash == record.task_spec_hash
    assert restored.model == record.model
    assert restored.mode == RunMode.direct
    assert restored.final_pass is True
    assert restored.artifact == record.artifact
    assert restored.schema_version == "compliance-eval-record/v1"
    assert restored.violations == []
    assert restored.repair_trace == []
    assert restored.error is None


def test_full_record_json_round_trip_preserves_nested_fields() -> None:
    record = _full_record()
    raw = record.model_dump_json()
    restored = ComplianceEvalRecord.model_validate_json(raw)

    assert len(restored.violations) == 1
    v = restored.violations[0]
    assert v.constraint_id == "c-format-1"
    assert v.level == ConstraintLevel.hard
    assert v.category == ConstraintCategory.format
    assert v.evidence_span is not None
    assert v.evidence_span.char_range == (0, 6)
    assert v.detected_at == ViolationStage.post_decode

    assert len(restored.repair_trace) == 1
    r = restored.repair_trace[0]
    assert r.round == 1
    assert r.ok is True
    assert r.strategy == RepairStrategy.patch

    assert restored.token_cost.prompt == 800
    assert restored.repair_rounds == 1
    assert abs(restored.latency_ms - 1342.7) < 0.01


def test_json_output_is_valid_json_and_contains_schema_version() -> None:
    raw = _minimal_record().model_dump_json()
    parsed = json.loads(raw)  # must not raise
    assert parsed["schema_version"] == "compliance-eval-record/v1"
    assert parsed["task_id"] == "task-001"


def test_record_with_error_field_round_trips() -> None:
    record = ComplianceEvalRecord(
        task_id="task-err",
        task_spec_hash="err-hash",
        model="gpt-4o-mini",
        mode=RunMode.prompt_retry,
        final_pass=False,
        artifact="",
        error=ComplianceError(
            kind="model_error",
            message="timeout after 30s",
            stage="generate",
        ),
    )
    restored = ComplianceEvalRecord.model_validate_json(record.model_dump_json())
    assert restored.error is not None
    assert restored.error.kind == "model_error"
    assert restored.error.stage == "generate"
    assert restored.final_pass is False


def test_round_trip_preserves_enum_string_values() -> None:
    record = ComplianceEvalRecord(
        task_id="t",
        task_spec_hash="h",
        model="m",
        mode=RunMode.full_pcl,
        final_pass=False,
        artifact="x",
    )
    raw_dict = json.loads(record.model_dump_json())
    assert raw_dict["mode"] == "full_pcl"
    restored = ComplianceEvalRecord.model_validate(raw_dict)
    assert restored.mode == RunMode.full_pcl


def test_multiple_records_survive_jsonl_round_trip(tmp_path) -> None:
    """Verify JSONL write→read cycle using evomerge.io helpers."""
    from evomerge.io import load_jsonl, write_jsonl

    records = [_minimal_record(), _full_record()]
    path = tmp_path / "compliance.jsonl"
    write_jsonl(records, path)

    loaded = load_jsonl(path, ComplianceEvalRecord)
    assert len(loaded) == 2
    assert loaded[0].task_id == "task-001"
    assert loaded[1].task_id == "task-002"
    assert len(loaded[1].violations) == 1
