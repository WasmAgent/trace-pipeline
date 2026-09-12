"""Adversarial and boundary tests for AEP validation.

Each test here is a hostile input an attacker (or a buggy producer) could
feed to the audit pipeline. The validator must survive everything and
produce a diagnosable result — never crash the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evomerge.validate.aep import validate_aep_file, validate_aep_record


def _base(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "aep/v0.5",
        "run_id": "run-adv-001",
        "created_at_ms": 1_757_460_000_000,
        "actions": [],
    }
    base.update(kwargs)
    return base


class TestHostileStructures:
    def test_deeply_nested_line_does_not_crash_the_run(self, tmp_path: Path) -> None:
        # 50k nested arrays blow jsonschema's recursive validator — the run
        # must classify the line as invalid instead of dying.
        line = '{"schema_version":"aep/v0.1","run_id":"x","created_at_ms":1,"actions":' + "[" * 50_000 + "]" * 50_000 + "}"
        p = tmp_path / "deep.jsonl"
        p.write_text(line)
        results = validate_aep_file(p)
        assert len(results) == 1
        assert results[0].valid_schema is False
        assert any("nesting too deep" in e or "internal validation failure" in e for e in results[0].errors)

    def test_proto_key_is_rejected(self) -> None:
        record = _base(__proto__={"admin": True})
        result = validate_aep_record(record)
        assert result.valid_schema is False
        assert any("__proto__" in e for e in result.errors)


class TestBoundaryInputs:
    def test_huge_string_field_is_handled(self) -> None:
        record = _base(user_id="A" * 10_000_000)
        result = validate_aep_record(record)  # must not crash
        assert result.run_id == "run-adv-001"

    def test_negative_timestamp_is_handled(self) -> None:
        # Canonical types created_at_ms as `number` with no minimum — the
        # validator must survive a negative value either way.
        result = validate_aep_record(_base(created_at_ms=-1))
        assert result.run_id == "run-adv-001"

    def test_thousand_action_record_validates(self) -> None:
        actions = [
            {
                "action_id": f"action-{i}",
                "tool_name": "bash",
                "state_changing": i % 2 == 0,
                "timestamp_ms": 1_757_460_000_000 + i,
                "result_digest": "sha256-x",
            }
            for i in range(1000)
        ]
        result = validate_aep_record(_base(actions=actions))
        assert result.passed


class TestAttributionFloorConsistency:
    def test_floor_weaker_than_observed_is_rejected(self) -> None:
        record = _base(
            run_attribution_backing_observed=["qualified_signature"],
            run_attribution_backing_floor="operator_asserted",
        )
        result = validate_aep_record(record)
        assert result.passed is False or any("weakest" in e for e in result.errors)

    def test_floor_not_present_in_observed_is_rejected(self) -> None:
        record = _base(
            run_attribution_backing_observed=["qualified_signature"],
            run_attribution_backing_floor="operator_asserted",
            run_attribution_backing_floor_copy="x",  # noqa: F841 - marker only
        )
        record["run_attribution_backing_floor"] = "unknown"
        result = validate_aep_record(record)
        assert any("canonical vocabulary" in e or "not present" in e for e in result.errors)

    def test_honest_floor_passes(self) -> None:
        record = _base(
            run_attribution_backing_observed=["operator_asserted", "qualified_signature"],
            run_attribution_backing_floor="operator_asserted",
        )
        result = validate_aep_record(record)
        assert result.passed
        assert not any("weakest" in e or "not present" in e for e in result.errors)


class TestJsonlRobustness:
    def test_unparseable_lines_are_classified_not_fatal(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            '{"schema_version":"aep/v0.1","run_id":"ok-1","created_at_ms":1}\n'
            "not json at all\n"
            '{"schema_version":"aep/v0.1","run_id":"ok-2","created_at_ms":2}\n'
        )
        results = validate_aep_file(p)
        # The hostile middle line produces its own error result; both valid
        # lines still validate.
        assert len(results) == 3
        assert not results[1].valid_schema
        assert results[0].passed and results[2].passed
