#!/usr/bin/env python3
"""Schema field drift checker.

Verifies that evomerge/schemas/ Pydantic models cover all fields declared in
the canonical JSON Schema files under wasmagent-js.

Usage:
    python scripts/check-schema-fields.py [--wasmagent-js PATH]

Exit codes:
    0  all fields present
    1  drift detected (fields missing or extra)
    2  canonical schema file not found (pass --wasmagent-js to fix)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DriftReport:
    schema_name: str
    canonical_fields: set[str]
    model_fields: set[str]

    @property
    def missing(self) -> set[str]:
        return self.canonical_fields - self.model_fields

    @property
    def extra(self) -> set[str]:
        return self.model_fields - self.canonical_fields

    @property
    def ok(self) -> bool:
        return not self.missing  # extra fields are acceptable (enrichment)


def _jsonschema_required_fields(schema: dict) -> set[str]:
    """Extract top-level required field names from a JSON Schema object."""
    required = set(schema.get("required", []))
    # also collect properties that appear in 'required' of any $defs entry
    for defn in schema.get("$defs", {}).values():
        required |= set(defn.get("required", []))
    return required


def _pydantic_fields(model_cls) -> set[str]:
    return set(model_cls.model_fields.keys())


def check_rollout_wire(wasmagent_js: Path | None) -> DriftReport | None:
    """Check RolloutBranchRecord against rollout-wire.schema.json."""
    from evomerge.schemas.rollout import RolloutBranchRecord

    if wasmagent_js is None:
        return None

    schema_path = (
        wasmagent_js
        / "packages/core/src/ranking/schemas/rollout-wire.schema.json"
    )
    if not schema_path.exists():
        print(f"[warn] not found: {schema_path}", file=sys.stderr)
        return None

    with open(schema_path) as fh:
        schema = json.load(fh)

    canonical = _jsonschema_required_fields(schema)
    model = _pydantic_fields(RolloutBranchRecord)
    return DriftReport("rollout-wire", canonical, model)


def check_training_record(wasmagent_js: Path | None) -> DriftReport | None:
    """Check DpoTrainingRecord against training-record.schema.json."""
    from evomerge.schemas.training import DpoTrainingRecord

    if wasmagent_js is None:
        return None

    schema_path = (
        wasmagent_js
        / "packages/core/src/ranking/schemas/training-record.schema.json"
    )
    if not schema_path.exists():
        print(f"[warn] not found: {schema_path}", file=sys.stderr)
        return None

    with open(schema_path) as fh:
        schema = json.load(fh)

    canonical = _jsonschema_required_fields(schema)
    model = _pydantic_fields(DpoTrainingRecord)
    return DriftReport("training-record", canonical, model)


def check_compliance_schema(wasmagent_js: Path | None) -> DriftReport | None:
    """Check ComplianceEvalRecord against compliance schema if present."""
    from evomerge.schemas.compliance import ComplianceEvalRecord

    if wasmagent_js is None:
        return None

    schema_path = (
        wasmagent_js / "packages/compliance/schemas/compliance-eval-record.schema.json"
    )
    if not schema_path.exists():
        return None  # optional schema — no warning if absent

    with open(schema_path) as fh:
        schema = json.load(fh)

    canonical = _jsonschema_required_fields(schema)
    model = _pydantic_fields(ComplianceEvalRecord)
    return DriftReport("compliance-eval-record", canonical, model)


def _standalone_checks() -> list[DriftReport]:
    """Field-presence checks that don't need the wasmagent-js repo."""
    reports: list[DriftReport] = []

    # rollout wire v1 minimum required fields (from data-loop-contract.md)
    CONTRACT_ROLLOUT_REQUIRED = {
        "schema_version", "rollout_id", "task", "branch_index",
        "temperature", "session_id", "tool_call_sequence", "final_answer",
    }
    from evomerge.schemas.rollout import RolloutBranchRecord
    reports.append(DriftReport(
        "rollout-wire (contract)",
        CONTRACT_ROLLOUT_REQUIRED,
        _pydantic_fields(RolloutBranchRecord),
    ))

    # training record minimum fields
    CONTRACT_TRAINING_REQUIRED = {"messages", "chosen", "rejected", "provenance"}
    from evomerge.schemas.training import DpoTrainingRecord
    reports.append(DriftReport(
        "training-record/dpo (contract)",
        CONTRACT_TRAINING_REQUIRED,
        _pydantic_fields(DpoTrainingRecord),
    ))

    CONTRACT_PPO_REQUIRED = {"messages", "reward", "provenance"}
    from evomerge.schemas.training import PpoTrainingRecord
    reports.append(DriftReport(
        "training-record/ppo (contract)",
        CONTRACT_PPO_REQUIRED,
        _pydantic_fields(PpoTrainingRecord),
    ))

    # compliance minimum fields
    CONTRACT_COMPLIANCE_REQUIRED = {
        "task_id", "task_spec_hash", "model", "mode",
        "final_pass", "artifact",
    }
    from evomerge.schemas.compliance import ComplianceEvalRecord
    reports.append(DriftReport(
        "compliance-eval-record (contract)",
        CONTRACT_COMPLIANCE_REQUIRED,
        _pydantic_fields(ComplianceEvalRecord),
    ))

    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--wasmagent-js", metavar="PATH",
        help="path to wasmagent-js repo root (enables JSON Schema comparison)",
    )
    ap.add_argument("--json", action="store_true", help="output JSON report")
    args = ap.parse_args()

    wasmagent_js = Path(args.wasmagent_js) if args.wasmagent_js else None

    reports: list[DriftReport] = _standalone_checks()

    for fn in (check_rollout_wire, check_training_record, check_compliance_schema):
        r = fn(wasmagent_js)
        if r is not None:
            reports.append(r)

    if args.json:
        out = [
            {
                "schema": r.schema_name,
                "ok": r.ok,
                "missing": sorted(r.missing),
                "extra": sorted(r.extra),
            }
            for r in reports
        ]
        print(json.dumps(out, indent=2))
    else:
        all_ok = True
        for r in reports:
            status = "OK" if r.ok else "DRIFT"
            print(f"  [{status}] {r.schema_name}")
            if r.missing:
                print(f"         MISSING: {sorted(r.missing)}")
                all_ok = False
            if r.extra:
                print(f"         extra (ok): {sorted(r.extra)}")
        if all_ok:
            print("\n✓ no schema drift detected")
        else:
            print("\n✗ schema drift detected — update evomerge/schemas/ to match")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
