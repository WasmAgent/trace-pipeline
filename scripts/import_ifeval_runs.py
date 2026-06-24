#!/usr/bin/env python3
"""Import wasmagent-js IFEval benchmark runs into evomerge training data.

Reads ComplianceEvalRecord JSONL files from the wasmagent-js benchmark
directories and exports:
  - compliance_sft.jsonl    (answerer + repairer SFT records)
  - compliance_dpo.jsonl    (repair-trace DPO pairs)
  - cross_mode_dpo.jsonl    (cross-mode preference pairs: full_pcl > direct etc.)
  - manifest.json           (counts + statistics)

Usage:
    python scripts/import_ifeval_runs.py \\
        --runs-dir /path/to/wasmagent-js/packages/compliance/benchmarks/ifeval \\
        --out-dir  data/training/ifeval

    # dry-run: print statistics without writing files
    python scripts/import_ifeval_runs.py \\
        --runs-dir /path/to/wasmagent-js/packages/compliance/benchmarks/ifeval \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Benchmark subdirectories to import (relative to --runs-dir)
BENCHMARK_DIRS = [
    ("results",                     "qwen2.5-1.5b", 42),
    ("results-seed43",              "qwen2.5-1.5b", 43),
    ("results-seed44",              "qwen2.5-1.5b", 44),
    ("results-llama-3.2-1b-seed42", "llama-3.2-1b",  42),
    ("results-llama-3.2-1b-seed43", "llama-3.2-1b",  43),
    ("results-llama-3.2-1b-seed44", "llama-3.2-1b",  44),
]


def load_runs(runs_dir: Path) -> list[dict]:
    """Load all ComplianceEvalRecord dicts from benchmark directories."""
    all_records: list[dict] = []
    for subdir, model_hint, seed in BENCHMARK_DIRS:
        jsonl = runs_dir / subdir / "runs.jsonl"
        if not jsonl.exists():
            print(f"[skip] not found: {jsonl}", file=sys.stderr)
            continue
        n = 0
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Annotate with seed for downstream analysis
                rec["_seed"] = seed
                all_records.append(rec)
                n += 1
        print(f"[load] {subdir}: {n} records")
    return all_records


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    modes    = Counter(r["mode"] for r in records)
    models   = Counter(r["model"] for r in records)
    n_pass   = sum(1 for r in records if r["final_pass"])
    n_repair = sum(1 for r in records if r.get("repair_rounds", 0) > 0)
    print(f"\n  total records : {len(records)}")
    print(f"  final_pass    : {n_pass} ({n_pass/len(records):.1%})")
    print(f"  with repair   : {n_repair}")
    print(f"  by mode       : {dict(modes)}")
    print(f"  by model      : {dict(models)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--runs-dir", required=True, metavar="DIR",
        help="path to wasmagent-js benchmarks/ifeval/ directory",
    )
    ap.add_argument(
        "--out-dir", default="data/training/ifeval", metavar="DIR",
        help="output directory for training JSONL files",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print statistics only, do not write files",
    )
    ap.add_argument(
        "--no-boundary", action="store_true",
        help="exclude boundary cases (prompt_retry beats full_pcl) from cross-mode DPO",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[error] runs-dir not found: {runs_dir}", file=sys.stderr)
        return 1

    # ── load ──────────────────────────────────────────────────────────────────
    print("loading records...")
    raw_records = load_runs(runs_dir)
    if not raw_records:
        print("[error] no records loaded", file=sys.stderr)
        return 1
    print_summary(raw_records)

    # ── parse into Pydantic models ────────────────────────────────────────────
    from evomerge.schemas.compliance import ComplianceEvalRecord
    records = []
    n_invalid = 0
    for raw in raw_records:
        raw.pop("_seed", None)  # strip annotation before Pydantic parse
        try:
            records.append(ComplianceEvalRecord.model_validate(raw))
        except Exception as exc:
            n_invalid += 1
            print(f"[warn] invalid record {raw.get('task_id')}: {exc}", file=sys.stderr)
    print(f"\n  parsed: {len(records)} valid, {n_invalid} invalid")

    # ── cross-mode DPO statistics ─────────────────────────────────────────────
    from evomerge.pipeline.cross_mode_dpo import cross_mode_dpo_records, cross_mode_summary
    stats = cross_mode_summary(records)
    print("\n  cross-mode pairing stats (per seed × model group):")
    for k, v in stats.items():
        print(f"    {k:<30} {v}")

    # ── convert ───────────────────────────────────────────────────────────────
    from evomerge.pipeline.compliance_sft import compliance_to_sft_records
    from evomerge.pipeline.compliance_dpo import compliance_to_dpo_records

    sft_records       = compliance_to_sft_records(records)
    repair_dpo        = compliance_to_dpo_records(records)
    cross_dpo         = cross_mode_dpo_records(
        records, include_boundary_cases=not args.no_boundary
    )

    # contamination check: count records with identical chosen/rejected
    n_invalid_dpo = sum(
        1 for r in repair_dpo + cross_dpo if r.chosen == r.rejected
    )

    print(f"\n  SFT records       : {len(sft_records)}")
    print(f"  repair DPO pairs  : {len(repair_dpo)}")
    print(f"  cross-mode DPO    : {len(cross_dpo)}")
    print(f"  invalid DPO pairs : {n_invalid_dpo}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    # ── write ─────────────────────────────────────────────────────────────────
    from evomerge.io import write_jsonl
    from evomerge.validate.schema_check import validate_training_record

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_schema_invalid = 0
    for rec_list in (sft_records, repair_dpo, cross_dpo):
        for rec in rec_list:
            if not validate_training_record(rec).ok:
                n_schema_invalid += 1

    files_written: dict[str, str] = {}

    if sft_records:
        p = out / "compliance_sft.jsonl"
        write_jsonl(sft_records, p)
        files_written["compliance_sft"] = str(p)
        print(f"\n  wrote {len(sft_records)} SFT records → {p}")

    if repair_dpo:
        p = out / "compliance_dpo.jsonl"
        write_jsonl(repair_dpo, p)
        files_written["compliance_dpo"] = str(p)
        print(f"  wrote {len(repair_dpo)} repair-DPO pairs → {p}")

    if cross_dpo:
        p = out / "cross_mode_dpo.jsonl"
        write_jsonl(cross_dpo, p)
        files_written["cross_mode_dpo"] = str(p)
        print(f"  wrote {len(cross_dpo)} cross-mode DPO pairs → {p}")

    manifest = {
        "source": str(runs_dir),
        "n_source_records": len(raw_records),
        "n_parsed": len(records),
        "n_invalid_source": n_invalid,
        "n_sft": len(sft_records),
        "n_repair_dpo": len(repair_dpo),
        "n_cross_mode_dpo": len(cross_dpo),
        "n_schema_invalid": n_schema_invalid,
        "cross_mode_stats": stats,
        "files": files_written,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  wrote manifest → {manifest_path}")

    print(f"\n✓ done — {len(sft_records) + len(repair_dpo) + len(cross_dpo)} total training records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
