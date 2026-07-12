#!/usr/bin/env python3
"""Sync canonical JSON Schema files from wasmagent-js into trace-pipeline.

The schema SSOT lives in wasmagent-js. This script copies the canonical
files into trace-pipeline/schemas/ and reports any field-level drift
between them and the current Pydantic models.

Usage:
    python scripts/sync-wasmagent-schemas.py --wasmagent-js /path/to/wasmagent-js
    python scripts/sync-wasmagent-schemas.py --wasmagent-js /path/to/wasmagent-js --check

Exit codes:
    0  in sync (or sync performed successfully)
    1  drift detected in --check mode
    2  canonical file not found
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"

# Maps wasmagent-js canonical path → local schemas/ filename
CANONICAL_MAP = {
    "packages/core/src/ranking/schemas/rollout-wire.schema.json": "rollout-wire.schema.json",
    "packages/core/src/ranking/schemas/training-record.schema.json": None,  # no direct mirror
}

# Fields that must be present in the Pydantic model for each canonical schema
REQUIRED_FIELD_COVERAGE = {
    "rollout-wire.schema.json": {
        "schema_version", "rollout_id", "task", "branch_index",
        "temperature", "session_id", "tool_call_sequence", "final_answer",
        "seed",
    },
}


def _pydantic_fields(model_cls) -> set[str]:
    return set(model_cls.model_fields.keys())


def check_field_coverage(canonical_schema: dict, model_fields: set[str], name: str) -> list[str]:
    """Return list of missing field names."""
    canonical_props = set(canonical_schema.get("properties", {}).keys())
    canonical_required = set(canonical_schema.get("required", []))
    missing = canonical_required - model_fields
    return sorted(missing)


def sync(wasmagent_js: Path, check_only: bool = False) -> int:
    if not wasmagent_js.exists():
        print(f"[error] wasmagent-js path not found: {wasmagent_js}", file=sys.stderr)
        return 2

    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    drift_found = False

    for canonical_rel, local_name in CANONICAL_MAP.items():
        if local_name is None:
            continue
        canonical_path = wasmagent_js / canonical_rel
        if not canonical_path.exists():
            print(f"[warn]  not found in wasmagent-js: {canonical_rel}", file=sys.stderr)
            continue

        canonical = json.loads(canonical_path.read_text())
        local_path = SCHEMAS_DIR / local_name

        if local_path.exists():
            local = json.loads(local_path.read_text())
            # Resolve oneOf/$defs structure: find RolloutBranchRecord definition
            if "properties" not in canonical and "$defs" in canonical:
                # Schema uses oneOf + $defs structure — find the primary record
                defs = canonical.get("$defs", {})
                if "RolloutBranchRecord" in defs:
                    canonical_resolved = defs["RolloutBranchRecord"]
                else:
                    # Fall back to first definition
                    canonical_resolved = next(iter(defs.values()), {})
            else:
                canonical_resolved = canonical

            canonical_props = canonical_resolved.get("properties", {})
            local_props     = local.get("properties", {})
            canonical_req   = set(canonical_resolved.get("required", []))
            local_req       = set(local.get("required", []))

            missing_props = set(canonical_props) - set(local_props)
            missing_req   = canonical_req - local_req

            if missing_props or missing_req:
                drift_found = True
                print(f"[drift] {local_name}")
                if missing_props:
                    print(f"        missing properties: {sorted(missing_props)}")
                if missing_req:
                    print(f"        missing required:   {sorted(missing_req)}")
            else:
                print(f"[ok]    {local_name}")
        else:
            print(f"[new]   {local_name} — not yet present locally")
            drift_found = True

        if not check_only:
            # Copy canonical, but preserve our $id and description
            merged = dict(canonical)
            if local_path.exists():
                existing = json.loads(local_path.read_text())
                merged["$id"]         = existing.get("$id", canonical.get("$id", ""))
                merged["description"] = existing.get("description", "")
            SCHEMAS_DIR.joinpath(local_name).write_text(
                json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"        synced → {local_name}")

    # Check Pydantic model field coverage against required contracts
    from evomerge.schemas.rollout import RolloutBranchRecord
    rollout_fields = _pydantic_fields(RolloutBranchRecord)
    for local_name, required in REQUIRED_FIELD_COVERAGE.items():
        missing = required - rollout_fields
        if missing:
            drift_found = True
            print(f"[model-drift] {local_name}: Pydantic model missing {sorted(missing)}")
        else:
            print(f"[model-ok]    {local_name}: Pydantic coverage complete")

    if drift_found and check_only:
        print("\n✗ drift detected", file=sys.stderr)
        return 1

    if not drift_found:
        print("\n✓ schemas in sync with wasmagent-js")
    else:
        print("\n✓ sync complete (with drift — review above)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wasmagent-js", required=True, metavar="PATH",
                    help="path to wasmagent-js repository root")
    ap.add_argument("--check", action="store_true",
                    help="check mode: report drift but do not copy files")
    args = ap.parse_args()
    return sync(Path(args.wasmagent_js), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
