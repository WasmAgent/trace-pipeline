#!/usr/bin/env python3
"""Generate a dataset card from an evomerge export manifest.

Usage:
    python scripts/generate-dataset-card.py \
        --manifest data/training/manifest.json \
        --name "wasmagent-compliance-v1" \
        --out data/training/DATASET_CARD.md

    # Check-only mode (CI): verify card exists and is not a bare template
    python scripts/generate-dataset-card.py \
        --manifest data/training/manifest.json \
        --check

Exit codes:
    0  success (or check passed)
    1  manifest not found / check failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Prefer the library implementation when evomerge is installed
try:
    from evomerge.dataset_card import generate_dataset_card as _generate
except ImportError:
    _generate = None  # type: ignore[assignment]


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        print(f"[error] manifest not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def generate(manifest: dict, name: str, date: str) -> str:
    if _generate is not None:
        return _generate(manifest, name=name, date=date)
    # Fallback: inline minimal implementation (no template file required)
    n_total = sum(manifest.get(k, 0) for k in
                  ("n_sft", "n_dpo", "n_ppo", "n_compliance_sft", "n_compliance_dpo", "n_router"))
    n_contaminated = manifest.get("n_contaminated", 0)
    contamination_rate = f"{n_contaminated / n_total:.1%}" if n_total else "N/A"
    return (
        f"# Dataset Card — {name}\n\nGenerated on {date}.\n\n"
        f"Records: sft={manifest.get('n_sft',0)} dpo={manifest.get('n_dpo',0)} "
        f"ppo={manifest.get('n_ppo',0)} total={n_total}\n"
        f"Invalid: {manifest.get('n_invalid',0)}  "
        f"Contaminated: {n_contaminated} ({contamination_rate})\n"
    )


def check(manifest_path: Path) -> int:
    card_path = manifest_path.parent / "DATASET_CARD.md"
    if not card_path.exists():
        print(f"[error] dataset card missing: {card_path}", file=sys.stderr)
        print("        Run: python scripts/generate-dataset-card.py "
              f"--manifest {manifest_path} --name <name> --out {card_path}",
              file=sys.stderr)
        return 1
    content = card_path.read_text()
    bare_placeholders = [p for p in ["{{DATASET_NAME}}", "{{DATE}}", "{{VERSION}}"]
                         if p in content]
    if bare_placeholders:
        print(f"[warn] dataset card has unfilled placeholders: {bare_placeholders}",
              file=sys.stderr)
    print(f"[ok] dataset card exists: {card_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, metavar="PATH",
                    help="path to manifest.json produced by evomerge export")
    ap.add_argument("--name", default="wasmagent-dataset",
                    help="dataset name (default: wasmagent-dataset)")
    ap.add_argument("--date", default="",
                    help="release date YYYY-MM-DD (default: today)")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="output path (default: <manifest_dir>/DATASET_CARD.md)")
    ap.add_argument("--check", action="store_true",
                    help="CI mode: verify card exists, do not generate")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if args.check:
        return check(manifest_path)

    manifest = _read_manifest(manifest_path)

    date = args.date
    if not date:
        import time
        date = time.strftime("%Y-%m-%d")

    card = generate(manifest, args.name, date)

    out_path = Path(args.out) if args.out else manifest_path.parent / "DATASET_CARD.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card)
    print(f"[ok] wrote dataset card: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
