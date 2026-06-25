"""AEP (Agent Evidence Protocol) record validation.

Validates AEP records against the JSON schema and checks evidence completeness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "aep-record.schema.json"


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as fh:
        return json.load(fh)


@dataclass
class AEPValidationResult:
    run_id: str
    valid_schema: bool
    has_model_id: bool
    has_actions: bool
    has_verifier_results: bool
    state_changing_actions_with_evidence: int
    state_changing_actions_total: int
    errors: list[str] = field(default_factory=list)

    @property
    def evidence_completeness(self) -> float:
        if self.state_changing_actions_total == 0:
            return 1.0
        return self.state_changing_actions_with_evidence / self.state_changing_actions_total

    @property
    def passed(self) -> bool:
        return self.valid_schema and len(self.errors) == 0


def validate_aep_record(record: dict[str, Any]) -> AEPValidationResult:
    run_id = record.get("run_id", "<unknown>")
    errors: list[str] = []

    # Schema validation
    valid_schema = True
    if _HAS_JSONSCHEMA:
        try:
            schema = _load_schema()
            jsonschema.validate(record, schema)
        except jsonschema.ValidationError as e:
            valid_schema = False
            errors.append(f"schema: {e.message}")
    else:
        # Minimal check without jsonschema
        if record.get("schema_version") != "aep/v0.1":
            valid_schema = False
            errors.append("schema_version must be 'aep/v0.1'")
        if "run_id" not in record:
            valid_schema = False
            errors.append("run_id is required")

    actions = record.get("actions", [])
    sc_actions = [a for a in actions if a.get("state_changing")]
    sc_with_evidence = [a for a in sc_actions if a.get("result_digest") or a.get("evidence_refs")]

    return AEPValidationResult(
        run_id=run_id,
        valid_schema=valid_schema,
        has_model_id=bool(record.get("model_id")),
        has_actions=len(actions) > 0,
        has_verifier_results=len(record.get("verifier_results", [])) > 0,
        state_changing_actions_total=len(sc_actions),
        state_changing_actions_with_evidence=len(sc_with_evidence),
        errors=errors,
    )


def validate_aep_file(path: Path) -> list[AEPValidationResult]:
    results = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                results.append(AEPValidationResult(
                    run_id=f"line-{i}",
                    valid_schema=False,
                    has_model_id=False,
                    has_actions=False,
                    has_verifier_results=False,
                    state_changing_actions_with_evidence=0,
                    state_changing_actions_total=0,
                    errors=[f"JSON parse error: {e}"],
                ))
                continue
            results.append(validate_aep_record(record))
    return results


def print_aep_report(results: list[AEPValidationResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"AEP validation: {passed}/{total} passed")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        ec = f"{r.evidence_completeness:.0%} evidence"
        print(f"  [{status}] {r.run_id} — {ec}")
        for err in r.errors:
            print(f"         error: {err}")
