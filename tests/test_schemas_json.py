"""Tests for schemas/ JSON Schema files and export-schemas.py."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"

EXPECTED_SCHEMAS = [
    "rollout-wire.schema.json",
    "task-spec.schema.json",
    "constraint-ir.schema.json",
    "constraint-violation.schema.json",
    "repair-trace-entry.schema.json",
    "compliance-eval-record.schema.json",
    "sft-training-record.schema.json",
    "dpo-training-record.schema.json",
    "ppo-training-record.schema.json",
]


class TestSchemaFilesExist:
    def test_all_expected_schema_files_present(self):
        for name in EXPECTED_SCHEMAS:
            path = SCHEMAS_DIR / name
            assert path.exists(), f"missing: {path}"

    def test_all_schema_files_are_valid_json(self):
        for name in EXPECTED_SCHEMAS:
            path = SCHEMAS_DIR / name
            data = json.loads(path.read_text())
            assert isinstance(data, dict), f"{name} is not a JSON object"

    def test_all_schemas_have_dollar_schema(self):
        for name in EXPECTED_SCHEMAS:
            data = json.loads((SCHEMAS_DIR / name).read_text())
            assert "$schema" in data, f"{name} missing $schema"
            assert "2020-12" in data["$schema"], f"{name} not draft 2020-12"

    def test_all_schemas_have_dollar_id(self):
        for name in EXPECTED_SCHEMAS:
            data = json.loads((SCHEMAS_DIR / name).read_text())
            assert "$id" in data, f"{name} missing $id"
            assert name in data["$id"], f"{name} $id does not contain filename"

    def test_all_schemas_have_description(self):
        for name in EXPECTED_SCHEMAS:
            data = json.loads((SCHEMAS_DIR / name).read_text())
            assert "description" in data, f"{name} missing description"
            assert len(data["description"]) > 10, f"{name} description too short"

    def test_all_schemas_have_properties(self):
        for name in EXPECTED_SCHEMAS:
            data = json.loads((SCHEMAS_DIR / name).read_text())
            assert "properties" in data or "$defs" in data, \
                f"{name} has neither properties nor $defs"


class TestSchemaContent:
    def _load(self, name: str) -> dict:
        return json.loads((SCHEMAS_DIR / name).read_text())

    def test_rollout_wire_has_required_fields(self):
        schema = self._load("rollout-wire.schema.json")
        required = set(schema.get("required", []))
        for field in ("rollout_id", "task", "branch_index", "final_answer"):
            assert field in required or field in schema.get("properties", {}), \
                f"rollout-wire missing field: {field}"

    def test_dpo_has_chosen_rejected(self):
        schema = self._load("dpo-training-record.schema.json")
        props = schema.get("properties", {})
        assert "chosen" in props
        assert "rejected" in props

    def test_ppo_has_reward(self):
        schema = self._load("ppo-training-record.schema.json")
        props = schema.get("properties", {})
        assert "reward" in props

    def test_compliance_eval_has_final_pass(self):
        schema = self._load("compliance-eval-record.schema.json")
        props = schema.get("properties", {})
        assert "final_pass" in props

    def test_task_spec_has_constraints(self):
        schema = self._load("task-spec.schema.json")
        props = schema.get("properties", {})
        assert "constraints" in props
        assert "intent" in props

    def test_sft_has_schema_version(self):
        schema = self._load("sft-training-record.schema.json")
        props = schema.get("properties", {})
        assert "schema_version" in props
        assert "messages" in props


class TestExportScript:
    def test_check_mode_passes_with_current_schemas(self):
        result = subprocess.run(
            [sys.executable, "scripts/export-schemas.py", "--check"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"export-schemas.py --check failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_generate_mode_produces_identical_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "scripts/export-schemas.py", "--out-dir", tmpdir],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            assert result.returncode == 0, result.stderr
            for name in EXPECTED_SCHEMAS:
                generated = json.loads((Path(tmpdir) / name).read_text())
                on_disk   = json.loads((SCHEMAS_DIR / name).read_text())
                assert generated == on_disk, f"{name} differs from generated output"

    def test_json_output_flag(self):
        result = subprocess.run(
            [sys.executable, "scripts/export-schemas.py", "--check", "--json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == len(EXPECTED_SCHEMAS)
        for entry in data:
            assert entry["ok"] is True
            assert entry["missing"] == []


class TestFixtureValidatesAgainstSchema:
    def test_fixture_fields_present_in_rollout_schema(self):
        schema = json.loads((SCHEMAS_DIR / "rollout-wire.schema.json").read_text())
        props = schema.get("properties", {})
        fixture = REPO_ROOT / "fixtures" / "data-loop" / "rollout-branches.v1.jsonl"
        with open(fixture) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for key in record:
                    assert key in props, \
                        f"fixture field '{key}' not in rollout-wire schema properties"
