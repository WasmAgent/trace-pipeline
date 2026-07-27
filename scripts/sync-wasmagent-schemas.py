#!/usr/bin/env python3
"""Check trace-pipeline's Pydantic models against the canonical schemas
published by the ``wasmagent-protocol`` package.

The schema SSOT is **WasmAgent/wasmagent-protocol** (PyPI: ``wasmagent-protocol``,
npm: ``@wasmagent/protocol``). trace-pipeline is a *consumer*: it must not fork
or hand-edit these schemas. This script no longer copies JSON out of
wasmagent-js — it reads the canonical schema straight from the installed package
and reports whether our Pydantic models still cover the required fields.

Usage:
    python scripts/sync-wasmagent-schemas.py            # report coverage
    python scripts/sync-wasmagent-schemas.py --check    # non-zero exit on drift

Exit codes:
    0  models cover the canonical required fields
    1  drift detected in --check mode
    2  wasmagent-protocol package not installed
"""
from __future__ import annotations

import argparse
import sys

# Map canonical schema id (in wasmagent-protocol) -> the Pydantic model in this
# repo expected to cover its required fields. Only schemas trace-pipeline
# actually consumes are listed.
COVERAGE = {
    "rollout-wire": ("evomerge.schemas.rollout", "RolloutBranchRecord"),
}


def _pydantic_fields(module_name: str, cls_name: str) -> set[str]:
    mod = __import__(module_name, fromlist=[cls_name])
    return set(getattr(mod, cls_name).model_fields.keys())


def _canonical_required(schema: dict) -> set[str]:
    """Required field names, resolving a oneOf/$defs primary-record layout."""
    if "properties" not in schema and "$defs" in schema:
        defs = schema["$defs"]
        primary = defs.get("RolloutBranchRecord") or next(iter(defs.values()), {})
        return set(primary.get("required", []))
    return set(schema.get("required", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a model is missing a canonical required field")
    # Accepted for backwards compatibility; the SSOT is now the package, so a
    # local wasmagent-js checkout is neither needed nor used.
    ap.add_argument("--wasmagent-js", metavar="PATH", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        from wasmagent_protocol import get_schema
    except ImportError:
        print("[error] wasmagent-protocol not installed. Run: pip install wasmagent-protocol",
              file=sys.stderr)
        return 2

    if args.wasmagent_js:
        print("[note]  --wasmagent-js is ignored; canonical schemas now come from "
              "the wasmagent-protocol package.", file=sys.stderr)

    drift = False
    for schema_id, (module_name, cls_name) in COVERAGE.items():
        try:
            schema = get_schema(schema_id)
        except KeyError:
            print(f"[error] {schema_id}: not found in wasmagent-protocol", file=sys.stderr)
            drift = True
            continue
        required = _canonical_required(schema)
        model_fields = _pydantic_fields(module_name, cls_name)
        missing = sorted(required - model_fields)
        if missing:
            drift = True
            print(f"[drift] {schema_id}: {cls_name} missing required {missing}")
        else:
            print(f"[ok]    {schema_id}: {cls_name} covers all canonical required fields")

    if drift and args.check:
        print("\n✗ drift detected against canonical wasmagent-protocol", file=sys.stderr)
        return 1
    print("\n✓ models consistent with canonical wasmagent-protocol"
          if not drift else "\n✓ report complete (drift above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
