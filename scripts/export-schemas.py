#!/usr/bin/env python3
"""Generate public JSON Schema files from evomerge Pydantic models.

Writes one .schema.json per model to schemas/.
Also verifies that the generated schemas validate the shared fixture.

Usage:
    python scripts/export-schemas.py          # generate + verify
    python scripts/export-schemas.py --check  # verify only (CI mode)

Exit codes:
    0  success
    1  schema/fixture mismatch or write failure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
FIXTURE_JSONL = REPO_ROOT / "fixtures" / "data-loop" / "rollout-branches.v1.jsonl"

SCHEMA_TARGETS = [
    # (module_path, class_name, output_filename, description)
    (
        "evomerge.schemas.rollout",
        "RolloutBranchRecord",
        "rollout-wire.schema.json",
        "One branch produced by wasmagent-js RolloutForkRunner. "
        "Schema version: rollout-wire/v1.",
    ),
    (
        "evomerge.schemas.compliance",
        "TaskSpec",
        "task-spec.schema.json",
        "Root contract for a WasmAgent compliance run, including constraints, "
        "tool policy, and repair configuration.",
    ),
    (
        "evomerge.schemas.compliance",
        "ConstraintIR",
        "constraint-ir.schema.json",
        "A single constraint with level, category, and optional repair policy.",
    ),
    (
        "evomerge.schemas.compliance",
        "ConstraintViolation",
        "constraint-violation.schema.json",
        "A detected constraint violation with location and stage information.",
    ),
    (
        "evomerge.schemas.compliance",
        "RepairTraceEntry",
        "repair-trace-entry.schema.json",
        "One round of repair: which violations were targeted and whether they resolved.",
    ),
    (
        "evomerge.schemas.compliance",
        "ComplianceEvalRecord",
        "compliance-eval-record.schema.json",
        "Final output of one WasmAgent compliance engine run.",
    ),
    (
        "evomerge.schemas.training",
        "SftTrainingRecord",
        "sft-training-record.schema.json",
        "Supervised fine-tuning record. schema_version: sft/v1.",
    ),
    (
        "evomerge.schemas.training",
        "DpoTrainingRecord",
        "dpo-training-record.schema.json",
        "DPO preference pair. schema_version: dpo/v1.",
    ),
    (
        "evomerge.schemas.training",
        "PpoTrainingRecord",
        "ppo-training-record.schema.json",
        "PPO/GRPO reward record. schema_version: ppo/v1.",
    ),
]

BASE_URI = "https://github.com/WasmAgent/trace-pipeline/blob/main/schemas/"


def _load_class(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


UPSTREAM_REQUIRED_OVERRIDES = {
    "rollout-wire.schema.json": ["tool_call_sequence"],
}


def generate_schemas(out_dir: Path) -> list[tuple[str, dict]]:
    """Generate JSON Schema dicts for all targets."""
    results = []
    for module_path, class_name, filename, description in SCHEMA_TARGETS:
        cls = _load_class(module_path, class_name)
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = BASE_URI + filename
        schema["description"] = description
        for field in UPSTREAM_REQUIRED_OVERRIDES.get(filename, []):
            if field not in schema.get("required", []):
                schema.setdefault("required", []).append(field)
        results.append((filename, schema))
    return results


def write_schemas(schemas: list[tuple[str, dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in schemas:
        path = out_dir / filename
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
        try:
            display = path.relative_to(REPO_ROOT)
        except ValueError:
            display = path
        print(f"  wrote {display}")


def verify_fixture(schemas: list[tuple[str, dict]]) -> bool:
    """Validate the shared fixture against rollout-wire schema."""
    rollout_schema = next(s for fn, s in schemas if fn == "rollout-wire.schema.json")
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        print("[skip] jsonschema not installed — skipping fixture validation")
        return True

    ok = True
    with open(FIXTURE_JSONL) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            try:
                jsonschema.validate(instance, rollout_schema)
            except jsonschema.ValidationError as exc:
                print(f"[fail] fixture line {lineno}: {exc.message}", file=sys.stderr)
                ok = False
    if ok:
        print("  fixture validates against rollout-wire schema")
    return ok


def check_existing(out_dir: Path, schemas: list[tuple[str, dict]]) -> bool:
    """Return True if on-disk schemas match generated schemas."""
    all_ok = True
    for filename, schema in schemas:
        path = out_dir / filename
        if not path.exists():
            print(f"[missing] {filename}", file=sys.stderr)
            all_ok = False
            continue
        on_disk = json.loads(path.read_text())
        if on_disk != schema:
            print(f"[drift]   {filename} — regenerate with export-schemas.py", file=sys.stderr)
            all_ok = False
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify mode: compare existing schemas to generated; exit 1 on mismatch",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="output JSON report (use with --check)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(SCHEMAS_DIR),
        help=f"output directory (default: {SCHEMAS_DIR})",
    )
    args = ap.parse_args()

    schemas = generate_schemas(Path(args.out_dir))

    if args.check:
        if getattr(args, "json", False):
            out = [
                {
                    "schema": fn,
                    "ok": (Path(args.out_dir) / fn).exists()
                        and json.loads((Path(args.out_dir) / fn).read_text()) == schema,
                    "missing": [],
                    "extra": [],
                }
                for fn, schema in schemas
            ]
            # mark missing/drifted entries
            for entry, (fn, schema) in zip(out, schemas):
                path = Path(args.out_dir) / fn
                if not path.exists():
                    entry["ok"] = False
                    entry["missing"] = list(schema.get("properties", {}).keys())
                elif json.loads(path.read_text()) != schema:
                    entry["ok"] = False
            print(json.dumps(out, indent=2))
            return 0 if all(e["ok"] for e in out) else 1

        ok = check_existing(Path(args.out_dir), schemas)
        if not ok:
            print("\nschema drift detected — run `python scripts/export-schemas.py` to fix")
            return 1
        print("\n✓ schemas are up to date")
        return 0

    write_schemas(schemas, Path(args.out_dir))
    ok = verify_fixture(schemas)
    print(f"\n✓ {len(schemas)} schemas written to {args.out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
