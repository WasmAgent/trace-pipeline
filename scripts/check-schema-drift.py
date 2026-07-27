#!/usr/bin/env python3
"""Drift gate: detect local schema files that re-fork canonical schemas.

Canonical schemas are owned by the ``wasmagent-protocol`` package (PyPI:
``wasmagent-protocol``).  This repo is a consumer — it must not fork them.

The gate scans ``schemas/*.schema.json`` and flags any file whose ``$id``
matches a canonical ``$id`` from the package, or whose filename matches a
canonical schema slug while using a non-repo ``$id``.

Usage:
    python scripts/check-schema-drift.py          # report only
    python scripts/check-schema-drift.py --check  # exit non-zero on drift

Exit codes:
    0  no drift
    1  re-forked canonical schema detected
    2  wasmagent-protocol not installed (gate cannot run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"

# Schemas regenerated from local Pydantic models use this ``$id`` prefix.
REPO_SCHEMA_PREFIX = "https://github.com/WasmAgent/trace-pipeline/blob/main/schemas/"

# Filename stem → canonical schema slug for schemas that were forked before
# the repo switched to consuming from the package.  These are tolerated with
# a warning; removing them is tracked separately (Milestone 1).
KNOWN_LEGACY_FORKS: dict[str, str] = {
    "aep-record": "aep-record",
}


def _load_canonical() -> tuple[set[str], set[str]]:
    """Return (canonical_$ids, canonical_slugs) from wasmagent-protocol."""
    from wasmagent_protocol import get_schema, schema_ids  # noqa: F401

    dollar_ids: set[str] = set()
    slugs: set[str] = set()
    for sid in schema_ids():
        schema = get_schema(sid)
        did = schema.get("$id", "")
        if did:
            dollar_ids.add(did)
        slugs.add(sid)
    return dollar_ids, slugs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if a new re-forked canonical schema is detected",
    )
    args = ap.parse_args()

    try:
        from wasmagent_protocol import get_schema, schema_ids  # noqa: F401
    except ImportError:
        print(
            "[error] wasmagent-protocol not installed — cannot run drift gate",
            file=sys.stderr,
        )
        return 2

    canonical_dollar_ids, canonical_slugs = _load_canonical()

    drift_found = False
    for schema_file in sorted(SCHEMAS_DIR.glob("*.schema.json")):
        data = json.loads(schema_file.read_text())
        local_id = data.get("$id", "")
        stem = schema_file.stem  # e.g. "aep-record" from "aep-record.schema.json"

        # Check 1: exact $id match against canonical
        if local_id in canonical_dollar_ids:
            if stem in KNOWN_LEGACY_FORKS:
                print(
                    f"[warn] {schema_file.name}: re-declares canonical $id "
                    f"(known legacy fork)"
                )
            else:
                print(
                    f"[drift] {schema_file.name}: re-declares canonical $id "
                    f"'{local_id}'"
                )
                drift_found = True
            continue

        # Check 2: filename matches canonical slug but $id is not repo-owned
        if stem in canonical_slugs and not local_id.startswith(REPO_SCHEMA_PREFIX):
            if stem in KNOWN_LEGACY_FORKS:
                print(
                    f"[warn] {schema_file.name}: filename matches canonical schema "
                    f"'{stem}' but uses non-repo $id '{local_id}' (known legacy fork)"
                )
            else:
                print(
                    f"[drift] {schema_file.name}: filename matches canonical schema "
                    f"'{stem}' but uses non-repo $id '{local_id}'"
                )
                drift_found = True
            continue

    if not drift_found:
        print("✓ no schema drift detected")
        return 0

    if args.check:
        print(
            "\n✗ schema drift detected — remove the forked file and "
            "consume from wasmagent-protocol instead",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
