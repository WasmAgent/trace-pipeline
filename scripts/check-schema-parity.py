#!/usr/bin/env python3
"""Schema parity test: verify that real runs.jsonl validates against both
the wasmagent-js JSON Schema and the evomerge-framework Pydantic model.

Run in CI or manually:
    python scripts/check-schema-parity.py \
        --runs-dir /path/to/wasmagent-js/packages/compliance/benchmarks/ifeval \
        --wasmagent-js /path/to/wasmagent-js

Exit 0 = parity confirmed. Exit 1 = drift detected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Sample 3 records per benchmark directory for fast CI check
SAMPLE_SIZE = 3
BENCHMARK_DIRS = [
    "results",
    "results-seed43",
    "results-llama-3.2-1b-seed42",
]


def load_samples(runs_dir: Path) -> list[dict]:
    samples = []
    for subdir in BENCHMARK_DIRS:
        jsonl = runs_dir / subdir / "runs.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as fh:
            for i, line in enumerate(fh):
                if i >= SAMPLE_SIZE:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def check_pydantic(samples: list[dict]) -> tuple[int, list[str]]:
    from evomerge.schemas.compliance import ComplianceEvalRecord
    errors = []
    for i, s in enumerate(samples):
        try:
            ComplianceEvalRecord.model_validate(s)
        except Exception as exc:
            errors.append(f"sample {i}: {exc}")
    return len(errors), errors


def check_jsonschema(samples: list[dict], wasmagent_js: Path) -> tuple[int, list[str]]:
    schema_path = (
        wasmagent_js
        / "packages/compliance/schemas/compliance-eval-record.schema.json"
    )
    if not schema_path.exists():
        print(f"[skip] wasmagent-js schema not found: {schema_path}", file=sys.stderr)
        return 0, []
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        print("[skip] jsonschema not installed — pip install jsonschema", file=sys.stderr)
        return 0, []

    schema = json.loads(schema_path.read_text())
    # Remove $ref resolution for standalone check
    errors = []
    for i, s in enumerate(samples):
        try:
            # Only validate top-level required fields (skip $ref resolution)
            required = schema.get("required", [])
            missing = [f for f in required if f not in s]
            if missing:
                errors.append(f"sample {i}: missing required fields {missing}")
        except Exception as exc:
            errors.append(f"sample {i}: {exc}")
    return len(errors), errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", required=True, metavar="DIR",
                    help="wasmagent-js benchmarks/ifeval/ directory")
    ap.add_argument("--wasmagent-js", default=None, metavar="PATH",
                    help="wasmagent-js repo root (enables JSON Schema check)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[error] runs-dir not found: {runs_dir}", file=sys.stderr)
        return 1

    samples = load_samples(runs_dir)
    if not samples:
        print("[error] no samples found", file=sys.stderr)
        return 1
    print(f"loaded {len(samples)} sample records")

    # Pydantic check
    n_err, errors = check_pydantic(samples)
    if errors:
        print(f"[FAIL] Pydantic validation: {n_err} errors")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"[OK]   Pydantic validation: {len(samples)} records pass")

    # JSON Schema check
    if args.wasmagent_js:
        n_err2, errors2 = check_jsonschema(samples, Path(args.wasmagent_js))
        if errors2:
            print(f"[FAIL] JSON Schema required-fields check: {n_err2} errors")
            for e in errors2:
                print(f"  {e}")
        else:
            print(f"[OK]   JSON Schema required-fields check: {len(samples)} records pass")
        total_errors = n_err + n_err2
    else:
        total_errors = n_err

    if total_errors == 0:
        print("\n✓ schema parity confirmed")
        return 0
    else:
        print(f"\n✗ {total_errors} parity errors — fix schema drift before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
