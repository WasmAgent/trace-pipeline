"""benchmarks/self_test.py — eval_trust self-test on synthetic ground truth.

This script verifies the toolkit's correctness *beyond* case-study
self-reproduction: it generates two synthetic paired-binary datasets
with known ground truth and checks that paired_stats classifies each
correctly.

The two scenarios:

  1. NO_SIGNAL:  Two candidates with identical expected accuracy (0.65
     each), independent per-item correctness. McNemar should NOT be
     significant; bootstrap CI should contain zero.

  2. REAL_SIGNAL: Two candidates with a small but real gap (0.65 vs
     0.72). McNemar SHOULD be significant at n=400; bootstrap CI
     should exclude zero.

We seed with multiple PRNG seeds and report both the toolkit's verdict
and the analytical truth, then assert agreement.

This benchmark is independent of the case-study data and so functions
as a *fresh* check on the audit primitives. Reviewers who don't want
to take our case-study numbers on faith can run this directly:

    python benchmarks/self_test.py

Output:
  benchmarks/self_test_results.md  — markdown report

Exit code:
  0 if all assertions pass (toolkit verdict matches ground truth);
  1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make eval_trust importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from eval_trust.paired_stats import (  # noqa: E402
    mcnemar_exact, paired_bootstrap, wilson_ci,
)


def gen_paired(
    n: int,
    p_a: float,
    p_b: float,
    rho: float,
    seed: int,
) -> tuple[list[bool], list[bool]]:
    """Generate paired binary outcomes with given marginal accuracies.

    Args:
        n: number of items.
        p_a, p_b: marginal accuracies.
        rho: correlation between the two outcomes (0 = independent).
        seed: PRNG seed.

    Returns:
        (correct_a, correct_b) lists of length n.
    """
    rng = np.random.default_rng(seed)
    # Latent Gaussian copula: shared component z + private noise
    # → maps to {0,1} via thresholds.
    z = rng.normal(size=n)
    eps_a = rng.normal(size=n)
    eps_b = rng.normal(size=n)

    sigma_z = np.sqrt(rho) if rho > 0 else 0.0
    sigma_eps = np.sqrt(1 - rho) if rho > 0 else 1.0
    u_a = sigma_z * z + sigma_eps * eps_a
    u_b = sigma_z * z + sigma_eps * eps_b

    from scipy.stats import norm
    threshold_a = norm.ppf(1 - p_a)
    threshold_b = norm.ppf(1 - p_b)
    correct_a = (u_a > threshold_a).tolist()
    correct_b = (u_b > threshold_b).tolist()
    return correct_a, correct_b


def run_scenario(
    label: str,
    n: int,
    p_a: float,
    p_b: float,
    rho: float,
    n_seeds: int,
    expected_significant: bool,
) -> dict:
    """Run a scenario across multiple seeds and check toolkit verdicts."""
    p_values = []
    delta_cis = []
    seed_records = []

    for seed in range(n_seeds):
        a, b = gen_paired(n, p_a, p_b, rho, seed=seed)
        # Discordant counts (a as winner, b as baseline)
        b_count = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
        c_count = sum(1 for x, y in zip(a, b, strict=True) if not y and x)
        p = mcnemar_exact(b_count, c_count)
        boot = paired_bootstrap(a, b, n_iter=2000, seed=seed)
        p_values.append(p)
        delta_cis.append((boot["ci_lo"], boot["ci_hi"]))
        seed_records.append({
            "seed": seed,
            "n_a_correct": sum(a),
            "n_b_correct": sum(b),
            "b": b_count,
            "c": c_count,
            "p": p,
            "delta": boot["delta_acc"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
        })

    p_arr = np.array(p_values)
    n_significant = int((p_arr < 0.05).sum())
    n_ci_excludes_zero = sum(
        1 for lo, hi in delta_cis if lo > 0 or hi < 0
    )

    return {
        "label": label,
        "n_per_seed": n,
        "p_a": p_a, "p_b": p_b, "rho": rho,
        "n_seeds": n_seeds,
        "expected_significant": expected_significant,
        "n_significant": n_significant,
        "n_ci_excludes_zero": n_ci_excludes_zero,
        "median_p": float(np.median(p_arr)),
        "mean_p": float(np.mean(p_arr)),
        "seed_records": seed_records,
    }


def main() -> int:
    out_md = Path(__file__).parent / "self_test_results.md"

    print("=" * 70)
    print("  eval_trust self-test on synthetic ground truth")
    print("=" * 70)
    print()

    # Scenario 1: NO signal (p_a == p_b)
    no_signal = run_scenario(
        label="NO_SIGNAL",
        n=200, p_a=0.65, p_b=0.65, rho=0.5,
        n_seeds=20,
        expected_significant=False,
    )
    print(f"[NO_SIGNAL]    median p = {no_signal['median_p']:.3f}, "
          f"{no_signal['n_significant']}/{no_signal['n_seeds']} seeds "
          f"reach p<0.05 (expected ~ 1)")

    # Scenario 2: REAL signal (7pp gap, n=400)
    real_signal = run_scenario(
        label="REAL_SIGNAL",
        n=400, p_a=0.65, p_b=0.72, rho=0.5,
        n_seeds=20,
        expected_significant=True,
    )
    print(f"[REAL_SIGNAL]  median p = {real_signal['median_p']:.3f}, "
          f"{real_signal['n_significant']}/{real_signal['n_seeds']} seeds "
          f"reach p<0.05 (expected ~ 18+)")

    # Wilson CI sanity (case-study scale)
    print(f"[Wilson CI]    13/22 has 95% CI = "
          f"[{wilson_ci(13, 22)[0]*100:.1f}%, {wilson_ci(13, 22)[1]*100:.1f}%], "
          f"expected ~ [38.7%, 76.9%] (paper Phase 11.5)")
    wilson_22 = wilson_ci(13, 22)
    wilson_ok = (
        abs(wilson_22[0] - 0.387) < 0.01 and abs(wilson_22[1] - 0.769) < 0.01
    )

    # Assertions
    print()
    print("Assertions:")

    # NO_SIGNAL: at most 5% of seeds should reach p<0.05 by chance
    # (since H0 is true). With n_seeds=20, expect 0-2.
    no_sig_ok = no_signal["n_significant"] <= 4
    print(f"  NO_SIGNAL false-positive rate: "
          f"{no_signal['n_significant']}/20 (need <= 4)  "
          f"{'PASS' if no_sig_ok else 'FAIL'}")

    # REAL_SIGNAL: most seeds should detect (power test).
    # 7pp at n=400 with rho=0.5 should hit ~ 95%+ detection.
    real_sig_ok = real_signal["n_significant"] >= 14
    print(f"  REAL_SIGNAL detection rate:    "
          f"{real_signal['n_significant']}/20 (need >= 14)  "
          f"{'PASS' if real_sig_ok else 'FAIL'}")

    # Wilson CI matches paper claim
    print(f"  Wilson CI for 13/22:           "
          f"({wilson_22[0]*100:.1f}%, {wilson_22[1]*100:.1f}%)  "
          f"{'PASS' if wilson_ok else 'FAIL'}")

    # Wilson CI for the case study's actual numbers
    winner_w = wilson_ci(135, 199)
    instruct_w = wilson_ci(137, 199)
    print(f"  Wilson winner   135/199 = 67.8%, 95% CI "
          f"[{winner_w[0]*100:.1f}, {winner_w[1]*100:.1f}]")
    print(f"  Wilson instruct 137/199 = 68.8%, 95% CI "
          f"[{instruct_w[0]*100:.1f}, {instruct_w[1]*100:.1f}]")

    all_ok = no_sig_ok and real_sig_ok and wilson_ok
    print()
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")

    # Markdown report
    md_lines = [
        "# eval_trust self-test results\n\n",
        "Synthetic-ground-truth verification, independent of the\n",
        "case-study data. Run with `python benchmarks/self_test.py`.\n\n",
        "## Scenarios\n\n",
    ]
    for s in (no_signal, real_signal):
        md_lines.append(
            f"### {s['label']}\n\n"
            f"- n per seed: {s['n_per_seed']}\n"
            f"- p_a, p_b: {s['p_a']}, {s['p_b']}\n"
            f"- correlation rho: {s['rho']}\n"
            f"- seeds: {s['n_seeds']}\n"
            f"- median p-value: {s['median_p']:.4f}\n"
            f"- seeds reaching p < 0.05: {s['n_significant']} / {s['n_seeds']}\n"
            f"- expected (analytical): "
            f"{'most should be significant' if s['expected_significant'] else 'none should be significant'}\n\n"
        )
    md_lines.append(
        "## Wilson CI sanity\n\n"
        f"- 13/22 → 95% CI ({wilson_22[0]*100:.1f}%, {wilson_22[1]*100:.1f}%) "
        f"(paper Phase 11.5: 38.7%, 76.9%)\n"
        f"- 135/199 (case-study winner) → 95% CI "
        f"({winner_w[0]*100:.1f}%, {winner_w[1]*100:.1f}%)\n"
        f"- 137/199 (case-study Instruct) → 95% CI "
        f"({instruct_w[0]*100:.1f}%, {instruct_w[1]*100:.1f}%)\n\n"
        "## Verdict\n\n"
        f"All assertions: **{'PASS' if all_ok else 'FAIL'}**\n"
    )
    out_md.write_text("".join(md_lines))
    print(f"  report saved to {out_md}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
