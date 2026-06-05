"""run_audit.py — End-to-end audit reproducer for the case study.

This single script reproduces the full case-study audit from the paper,
using only the raw logs in ``data/`` and the toolkit in ``eval_trust/``.

Usage:
    python run_audit.py
    python run_audit.py --json out.json   # save full report

What it does:
    1. Loads paired greedy-eval logs from data/ (winner + Instruct, n=199).
    2. Computes paired McNemar exact two-sided p (the headline test).
    3. Wilson 95% CI on each candidate's accuracy.
    4. Paired bootstrap 95% CI on the delta.
    5. Reports the SC-5 lottery rate from data/self_consistency_full.json.
    6. Reports the T0v2 channel breakdown from data/t0v2_aggregate.json.
    7. Prints a verdict ("real / lottery / primitive smell / reasoning bottleneck").

Output:
    Plain-text report to stdout. Optional --json to dump structured data.

Tested on Python 3.10+ with stdlib only (numpy/scipy optional).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make eval_trust importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from eval_trust.paired_stats import (  # noqa: E402
    mcnemar_exact, paired_bootstrap, wilson_ci,
)


def audit_pair(winner_path: Path, baseline_path: Path) -> dict:
    """Run the paired audit on two greedy-eval JSON logs."""
    with open(winner_path) as f:
        winner = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    wm = {r["id"]: r["correct"] for r in winner["results"] if "correct" in r}
    bm = {r["id"]: r["correct"] for r in baseline["results"] if "correct" in r}
    common = sorted(set(wm) & set(bm))
    n = len(common)

    n_winner = sum(wm[i] for i in common)
    n_baseline = sum(bm[i] for i in common)

    b = sum(1 for i in common if bm[i] and not wm[i])  # baseline-only correct
    c = sum(1 for i in common if not bm[i] and wm[i])  # winner-only correct

    p = mcnemar_exact(b, c)
    w_lo, w_hi = wilson_ci(n_winner, n)
    b_lo, b_hi = wilson_ci(n_baseline, n)

    boot = paired_bootstrap(
        [wm[i] for i in common],
        [bm[i] for i in common],
        n_iter=10000, seed=42,
    )

    return {
        "n": n,
        "winner_correct": n_winner,
        "baseline_correct": n_baseline,
        "winner_acc": n_winner / n,
        "baseline_acc": n_baseline / n,
        "delta_pp": (n_winner - n_baseline) / n * 100,
        "winner_wilson_95ci": [w_lo, w_hi],
        "baseline_wilson_95ci": [b_lo, b_hi],
        "b_baseline_only": b,
        "c_winner_only": c,
        "mcnemar_2sided_p": p,
        "paired_bootstrap_delta_pp_95ci": [
            boot["ci_lo"] * 100, boot["ci_hi"] * 100,
        ],
        "paired_bootstrap_p_two_sided": boot["p_two_sided"],
    }


def lottery_rate_from(sc_path: Path) -> dict:
    with open(sc_path) as f:
        d = json.load(f)
    return {
        "n_tested": d.get("n_tested"),
        "lottery_count": d.get("lottery_count"),
        "lottery_rate": d.get("lottery_rate"),
    }


def t0v2_channels_from(agg_path: Path) -> dict:
    with open(agg_path) as f:
        d = json.load(f)
    return {
        "n_total": d.get("n_total"),
        "n_wrong": d.get("n_wrong"),
        "n_first_class": d.get("n_first_class"),
        "first_class_rate_of_total": d.get("first_class_rate_of_total"),
        "verdict_counts": d.get("verdict_counts"),
        "decision": d.get("decision_v2", {}).get("case"),
    }


def print_report(report: dict) -> None:
    p = report["paired"]
    sc = report["lottery"]
    t = report["t0v2"]

    print("=" * 70)
    print("  eval_trust audit — case-study reproduction")
    print("=" * 70)
    print()
    print(f"  Paired McNemar (n={p['n']}):")
    print(f"    winner   {p['winner_correct']}/{p['n']} = {p['winner_acc']*100:5.1f}%  "
          f"95% CI [{p['winner_wilson_95ci'][0]*100:.1f}, {p['winner_wilson_95ci'][1]*100:.1f}]")
    print(f"    baseline {p['baseline_correct']}/{p['n']} = {p['baseline_acc']*100:5.1f}%  "
          f"95% CI [{p['baseline_wilson_95ci'][0]*100:.1f}, {p['baseline_wilson_95ci'][1]*100:.1f}]")
    print(f"    delta    {p['delta_pp']:+5.1f} pp  bootstrap 95% CI "
          f"[{p['paired_bootstrap_delta_pp_95ci'][0]:+.1f}, "
          f"{p['paired_bootstrap_delta_pp_95ci'][1]:+.1f}]")
    print(f"    b={p['b_baseline_only']}, c={p['c_winner_only']}, "
          f"McNemar exact 2-sided p = {p['mcnemar_2sided_p']:.4f}")
    print()

    print(f"  Self-consistency lottery (SC-5 majority on {sc['n_tested']} greedy-wrong):")
    print(f"    lottery: {sc['lottery_count']}/{sc['n_tested']} = "
          f"{sc['lottery_rate']*100:.1f}%")
    print()

    print(f"  T0v2 channel triage (n_total={t['n_total']}, n_wrong={t['n_wrong']}):")
    for ch, n_ch in (t["verdict_counts"] or {}).items():
        print(f"    {ch:20s}  {n_ch}")
    print(f"    first-class rate (of total) = "
          f"{t['first_class_rate_of_total']*100:.1f}%")
    print(f"    aggregator decision: {t['decision']}")
    print()

    # Verdict
    verdict_parts = []
    if p["mcnemar_2sided_p"] > 0.10:
        verdict_parts.append("paired McNemar non-significant")
    elif p["mcnemar_2sided_p"] < 0.05:
        verdict_parts.append("paired-significant")
    else:
        verdict_parts.append("paired marginal (0.05<=p<=0.10)")

    if sc["lottery_rate"] >= 0.30:
        verdict_parts.append("HIGH lottery (>=30%): switch to SC-5 before claiming")
    elif sc["lottery_rate"] >= 0.15:
        verdict_parts.append(f"moderate lottery ({sc['lottery_rate']*100:.0f}%)")

    if t["first_class_rate_of_total"] and t["first_class_rate_of_total"] >= 0.15:
        verdict_parts.append("alpha verdict: surface artefacts present (consider repair)")
    else:
        verdict_parts.append("gamma verdict: reasoning bottleneck dominates")

    print(f"  VERDICT: {' | '.join(verdict_parts)}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--winner", default="data/winner_max_new768.json")
    ap.add_argument("--baseline", default="data/instruct_max_new768.json")
    ap.add_argument("--sc", default="data/self_consistency_full.json")
    ap.add_argument("--t0v2", default="data/t0v2_aggregate.json")
    ap.add_argument("--json", help="optional path to dump full report as JSON")
    args = ap.parse_args()

    report = {
        "paired": audit_pair(Path(args.winner), Path(args.baseline)),
        "lottery": lottery_rate_from(Path(args.sc)),
        "t0v2": t0v2_channels_from(Path(args.t0v2)),
    }
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"  full report dumped to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
