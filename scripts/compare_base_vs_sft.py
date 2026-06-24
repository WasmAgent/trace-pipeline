#!/usr/bin/env python3
"""Compare base model vs fine-tuned model across 3 seeds on IFEval.

Reads:
  base:     packages/compliance/benchmarks/ifeval/results{,-seed43,-seed44}/runs.jsonl
  fine-tuned: packages/compliance/benchmarks/ifeval/results-sft-seed{42,43,44}/runs.jsonl

Computes per-seed and aggregate pass rates, McNemar p, Wilson CIs,
and writes a JSON report + updates papers/compliance_model/report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
WASMAGENT   = REPO_ROOT.parent / "wasmagent-js"
IFEVAL_DIR  = WASMAGENT / "packages/compliance/benchmarks/ifeval"

BASE_DIRS = ["results", "results-seed43", "results-seed44"]
SFT_DIRS  = ["results-sft-seed42", "results-sft-seed43", "results-sft-seed44"]
SEEDS     = [42, 43, 44]

sys.path.insert(0, str(REPO_ROOT))
from eval_trust.paired_stats import mcnemar_exact, paired_bootstrap, wilson_ci


def load_by_mode(path: Path) -> dict[tuple, dict]:
    records = {}
    if not path.exists():
        return records
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[(r["task_id"], r["mode"])] = r
    return records


def pass_rate(records: dict, mode: str) -> tuple[int, int]:
    recs = [v for (tid, m), v in records.items() if m == mode]
    n_pass = sum(1 for r in recs if r["final_pass"])
    return n_pass, len(recs)


def paired_mcnemar(base: dict, sft: dict, mode: str) -> dict:
    common = sorted(
        set(tid for (tid, m) in base if m == mode) &
        set(tid for (tid, m) in sft  if m == mode)
    )
    if not common:
        return {"n": 0, "b": 0, "c": 0, "p": 1.0}
    bp = [base[(tid, mode)]["final_pass"] for tid in common]
    sp = [sft[(tid, mode)]["final_pass"]  for tid in common]
    b = sum(1 for x, y in zip(bp, sp) if x and not y)
    c = sum(1 for x, y in zip(bp, sp) if not x and y)
    return {
        "n": len(common),
        "b": b, "c": c,
        "p": mcnemar_exact(b, c),
        "base_pass": sum(bp),
        "sft_pass":  sum(sp),
    }


def main():
    rows = []
    all_base_direct = []
    all_sft_direct  = []
    all_base_pcl    = []
    all_sft_pcl     = []

    for seed, bdir, sdir in zip(SEEDS, BASE_DIRS, SFT_DIRS):
        base = load_by_mode(IFEVAL_DIR / bdir / "runs.jsonl")
        sft  = load_by_mode(IFEVAL_DIR / sdir / "runs.jsonl")

        if not base or not sft:
            print(f"[skip] seed {seed}: missing data (base={bool(base)}, sft={bool(sft)})",
                  file=sys.stderr)
            continue

        b_dir_n, b_dir_tot = pass_rate(base, "direct")
        s_dir_n, s_dir_tot = pass_rate(sft,  "direct")
        b_pcl_n, b_pcl_tot = pass_rate(base, "full_pcl")
        s_pcl_n, s_pcl_tot = pass_rate(sft,  "full_pcl")

        mc_dir = paired_mcnemar(base, sft, "direct")
        mc_pcl = paired_mcnemar(base, sft, "full_pcl")

        row = {
            "seed": seed,
            "base_direct":   round(b_dir_n / b_dir_tot * 100, 1) if b_dir_tot else None,
            "sft_direct":    round(s_dir_n / s_dir_tot * 100, 1) if s_dir_tot else None,
            "base_full_pcl": round(b_pcl_n / b_pcl_tot * 100, 1) if b_pcl_tot else None,
            "sft_full_pcl":  round(s_pcl_n / s_pcl_tot * 100, 1) if s_pcl_tot else None,
            "mcnemar_direct": mc_dir,
            "mcnemar_full_pcl": mc_pcl,
        }
        rows.append(row)
        print(f"seed {seed}: base direct={row['base_direct']}%  sft direct={row['sft_direct']}%  "
              f"base pcl={row['base_full_pcl']}%  sft pcl={row['sft_full_pcl']}%  "
              f"McNemar_pcl p={mc_pcl['p']:.4f}")

        # accumulate for aggregate
        common_dir = sorted(set(tid for (tid, m) in base if m=="direct") &
                            set(tid for (tid, m) in sft  if m=="direct"))
        all_base_direct.extend(base[(tid,"direct")]["final_pass"] for tid in common_dir)
        all_sft_direct.extend(sft[(tid,"direct")]["final_pass"]   for tid in common_dir)
        common_pcl = sorted(set(tid for (tid, m) in base if m=="full_pcl") &
                            set(tid for (tid, m) in sft  if m=="full_pcl"))
        all_base_pcl.extend(base[(tid,"full_pcl")]["final_pass"] for tid in common_pcl)
        all_sft_pcl.extend(sft[(tid,"full_pcl")]["final_pass"]   for tid in common_pcl)

    if not rows:
        print("No data loaded — exiting", file=sys.stderr)
        return 1

    # Aggregate
    n = len(all_base_pcl)
    agg_b_pcl = sum(all_base_pcl) / n if n else 0
    agg_s_pcl = sum(all_sft_pcl)  / n if n else 0
    b_agg = sum(1 for x, y in zip(all_base_pcl, all_sft_pcl) if x and not y)
    c_agg = sum(1 for x, y in zip(all_base_pcl, all_sft_pcl) if not x and y)
    p_agg = mcnemar_exact(b_agg, c_agg)
    boot  = paired_bootstrap(all_base_pcl, all_sft_pcl, n_iter=10000)
    ci_base = wilson_ci(sum(all_base_pcl), n)
    ci_sft  = wilson_ci(sum(all_sft_pcl),  n)

    print(f"\nAggregate (n={n}, {len(rows)} seeds):")
    print(f"  base full_pcl : {agg_b_pcl*100:.1f}%  CI [{ci_base[0]*100:.1f}%, {ci_base[1]*100:.1f}%]")
    print(f"  sft  full_pcl : {agg_s_pcl*100:.1f}%  CI [{ci_sft[0]*100:.1f}%, {ci_sft[1]*100:.1f}%]")
    print(f"  delta         : {(agg_s_pcl-agg_b_pcl)*100:+.1f}pp")
    print(f"  McNemar b={b_agg} c={c_agg} p={p_agg:.4f}")
    print(f"  bootstrap CI  : [{boot['ci_lo']*100:.1f}%, {boot['ci_hi']*100:.1f}%]")

    report = {
        "per_seed": rows,
        "aggregate": {
            "n_paired": n,
            "n_seeds":  len(rows),
            "base_full_pcl_mean": round(agg_b_pcl * 100, 2),
            "sft_full_pcl_mean":  round(agg_s_pcl * 100, 2),
            "delta_pp": round((agg_s_pcl - agg_b_pcl) * 100, 2),
            "mcnemar_b": b_agg, "mcnemar_c": c_agg, "mcnemar_p": round(p_agg, 6),
            "significant_05": p_agg < 0.05,
            "bootstrap_ci_lo": round(boot["ci_lo"] * 100, 2),
            "bootstrap_ci_hi": round(boot["ci_hi"] * 100, 2),
            "wilson_ci_base": [round(ci_base[0]*100, 1), round(ci_base[1]*100, 1)],
            "wilson_ci_sft":  [round(ci_sft[0]*100, 1),  round(ci_sft[1]*100, 1)],
        },
    }

    out = REPO_ROOT / "data/eval/group_ac_3seed_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
