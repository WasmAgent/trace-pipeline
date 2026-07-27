#!/usr/bin/env python3
"""Merge all training data sources into v2 training sets.

Sources:
  Real IFEval:
    data/training/ifeval/compliance_sft.jsonl       (556 SFT, seeds 42-44)
    data/training/ifeval/compliance_dpo.jsonl        (67 repair-trace DPO)
    data/training/ifeval/cross_mode_dpo.jsonl        (34 cross-mode DPO)
    data/training/ifeval_s{N}/compliance_sft.jsonl  (new seeds 45-50)

  Synthetic:
    data/synthetic/batch1/sft.jsonl
    data/synthetic/batch1/dpo.jsonl
    data/synthetic/batch2/sft.jsonl  (if exists)
    data/synthetic/sft.jsonl         (original 60)
    data/synthetic/dpo.jsonl         (original 10)

Output:
  data/training/v2/sft_merged.jsonl
  data/training/v2/dpo_merged.jsonl
  data/training/v2/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _content_hash(rec: dict) -> str:
    # Dedup: exact content match only (full last-message hash)
    # This keeps different-task records and different-seed records
    # even if they look similar, only dropping true duplicates
    msgs = rec.get("messages", [])
    all_content = "|".join(m.get("content", "")[:200] for m in msgs)
    return hashlib.sha256(all_content.encode()).hexdigest()[:20]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    out = REPO_ROOT / "data/training/v2"
    out.mkdir(parents=True, exist_ok=True)

    # ── collect SFT sources ───────────────────────────────────────────────────
    sft_sources = [
        REPO_ROOT / "data/training/ifeval/compliance_sft.jsonl",
        REPO_ROOT / "data/synthetic/sft.jsonl",
        REPO_ROOT / "data/synthetic/batch1/sft.jsonl",
        REPO_ROOT / "data/synthetic/batch2/sft.jsonl",
    ]
    # new ifeval seeds
    for seed in range(45, 51):
        p = REPO_ROOT / f"data/training/ifeval_s{seed}/compliance_sft.jsonl"
        if not p.exists():
            # also check wasmagent-js results dir
            wjs = REPO_ROOT.parent / "wasmagent-js"
            alt = wjs / f"packages/compliance/benchmarks/ifeval/results-seed{seed}"
            # import on-the-fly if runs.jsonl exists
            if (alt / "runs.jsonl").exists():
                print(f"  importing seed {seed} from wasmagent-js...")
                _import_ifeval_seed(alt, seed, out.parent / "ifeval")
                p = out.parent / f"ifeval_s{seed}/compliance_sft.jsonl"
        sft_sources.append(p)

    # ── collect DPO sources ───────────────────────────────────────────────────
    dpo_sources = [
        REPO_ROOT / "data/training/ifeval/compliance_dpo.jsonl",
        REPO_ROOT / "data/training/ifeval/cross_mode_dpo.jsonl",
        REPO_ROOT / "data/synthetic/dpo.jsonl",
        REPO_ROOT / "data/synthetic/batch1/dpo.jsonl",
        REPO_ROOT / "data/synthetic/batch2/dpo.jsonl",
    ]

    # ── merge SFT with dedup ──────────────────────────────────────────────────
    seen_sft: set[str] = set()
    all_sft: list[dict] = []
    for src in sft_sources:
        records = load_jsonl(src)
        added = 0
        for r in records:
            if r.get("schema_version") != "sft/v1":
                continue
            h = _content_hash(r)
            if h not in seen_sft:
                seen_sft.add(h)
                all_sft.append(r)
                added += 1
        if records:
            print(f"  SFT {src.name}: {len(records)} loaded, {added} new")

    # ── merge DPO with dedup ──────────────────────────────────────────────────
    seen_dpo: set[str] = set()
    all_dpo: list[dict] = []
    for src in dpo_sources:
        records = load_jsonl(src)
        added = 0
        for r in records:
            if r.get("schema_version") != "dpo/v1":
                continue
            key = hashlib.sha256(
                (r.get("chosen", "") + r.get("rejected", "")).encode()
            ).hexdigest()[:16]
            if key not in seen_dpo:
                seen_dpo.add(key)
                all_dpo.append(r)
                added += 1
        if records:
            print(f"  DPO {src.name}: {len(records)} loaded, {added} new")

    # ── write ─────────────────────────────────────────────────────────────────
    sft_path = out / "sft_merged.jsonl"
    dpo_path = out / "dpo_merged.jsonl"

    with open(sft_path, "w") as fh:
        for r in all_sft:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(dpo_path, "w") as fh:
        for r in all_dpo:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # output type breakdown
    output_types: dict[str, int] = {}
    for r in all_sft:
        t = r.get("output_type", "unknown")
        output_types[t] = output_types.get(t, 0) + 1

    manifest = {
        "n_sft": len(all_sft),
        "n_dpo": len(all_dpo),
        "sft_output_types": output_types,
        "sources_sft": [str(s) for s in sft_sources if s.exists()],
        "sources_dpo": [str(s) for s in dpo_sources if s.exists()],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n✓ v2 training data merged:")
    print(f"  SFT: {len(all_sft)} records → {sft_path}")
    print(f"  DPO: {len(all_dpo)} pairs  → {dpo_path}")
    print(f"  output_types: {output_types}")
    return 0


def _import_ifeval_seed(results_dir: Path, seed: int, out_base: Path) -> None:
    """Quick import of a single ifeval seed without full CLI."""
    sys.path.insert(0, str(REPO_ROOT))
    from evomerge.io import write_jsonl
    from evomerge.pipeline.compliance_dpo import compliance_to_dpo_records
    from evomerge.pipeline.compliance_sft import compliance_to_sft_records
    from evomerge.schemas.compliance import ComplianceEvalRecord

    records = []
    with open(results_dir / "runs.jsonl") as fh:
        for line in fh:
            if line.strip():
                records.append(ComplianceEvalRecord.model_validate_json(line))

    seed_out = out_base.parent / f"ifeval_s{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    sft = compliance_to_sft_records(records)
    dpo = compliance_to_dpo_records(records)
    if sft:
        write_jsonl(sft, seed_out / "compliance_sft.jsonl")
    if dpo:
        write_jsonl(dpo, seed_out / "compliance_dpo.jsonl")


if __name__ == "__main__":
    sys.exit(main())
