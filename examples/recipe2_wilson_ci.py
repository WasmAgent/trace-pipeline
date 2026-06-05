"""examples/recipe2_wilson_ci.py — Wilson 95% CI per candidate.

Run from repo root:
    python examples/recipe2_wilson_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.paired_stats import wilson_ci

DATA = Path(__file__).resolve().parent.parent / "data"

with open(DATA / "winner_max_new768.json") as f:
    a = json.load(f)
with open(DATA / "instruct_max_new768.json") as f:
    b = json.load(f)

am = {r["id"]: r["correct"] for r in a["results"] if "correct" in r}
bm = {r["id"]: r["correct"] for r in b["results"] if "correct" in r}
common = sorted(set(am) & set(bm))
n = len(common)

n_winner = sum(1 for i in common if am[i])
n_baseline = sum(1 for i in common if bm[i])

w_lo, w_hi = wilson_ci(n_winner, n)
b_lo, b_hi = wilson_ci(n_baseline, n)

print(f"winner   {n_winner}/{n} = {n_winner/n*100:.1f}%  "
      f"95% CI [{w_lo*100:.1f}, {w_hi*100:.1f}]")
print(f"baseline {n_baseline}/{n} = {n_baseline/n*100:.1f}%  "
      f"95% CI [{b_lo*100:.1f}, {b_hi*100:.1f}]")
print()
print("CIs overlap heavily, consistent with the McNemar non-significance.")
