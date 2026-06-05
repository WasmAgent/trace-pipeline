"""examples/recipe3_paired_bootstrap.py — bootstrap CI on the delta.

Run from repo root:
    python examples/recipe3_paired_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.paired_stats import paired_bootstrap

DATA = Path(__file__).resolve().parent.parent / "data"

with open(DATA / "winner_max_new768.json") as f:
    a = json.load(f)
with open(DATA / "instruct_max_new768.json") as f:
    b = json.load(f)

am = {r["id"]: r["correct"] for r in a["results"] if "correct" in r}
bm = {r["id"]: r["correct"] for r in b["results"] if "correct" in r}
common = sorted(set(am) & set(bm))

a_arr = [am[i] for i in common]
b_arr = [bm[i] for i in common]
out = paired_bootstrap(a_arr, b_arr, n_iter=10000, seed=42)

print(f"delta = {out['delta_acc']*100:+.2f} pp")
print(f"95% CI: [{out['ci_lo']*100:+.2f}, {out['ci_hi']*100:+.2f}] pp")
print(f"two-sided p = {out['p_two_sided']:.3f}")
print()
print("The CI contains zero — same finding as McNemar's p > 0.05, in CI form.")
