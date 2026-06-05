"""examples/recipe1_paired_mcnemar.py — the headline test.

You have two greedy-eval JSON logs from candidates A and B, evaluated
on the same item set. Question: is A's accuracy advantage real?

Run from repo root:
    python examples/recipe1_paired_mcnemar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest

DATA = Path(__file__).resolve().parent.parent / "data"

with open(DATA / "winner_max_new768.json") as f:
    a = json.load(f)
with open(DATA / "instruct_max_new768.json") as f:
    b = json.load(f)

am = {r["id"]: r["correct"] for r in a["results"] if "correct" in r}
bm = {r["id"]: r["correct"] for r in b["results"] if "correct" in r}
common = sorted(set(am) & set(bm))

# Discordant cells: McNemar's test summands
b_count = sum(1 for i in common if bm[i] and not am[i])  # B-only correct
c_count = sum(1 for i in common if not bm[i] and am[i])  # A-only correct

p = binomtest(min(b_count, c_count), b_count + c_count,
              p=0.5, alternative="two-sided").pvalue

print(f"AUDITED  (max_new=768): n={len(common)}  b={b_count}  c={c_count}  p={p:.4f}")
print( "ORIGINAL (max_new=300): n=200  b=21  c=41  p=0.0151   <-- was paired-significant")
print()
print("With b=29, c=27 nearly equal, the paired difference is not significant.")
print("The +10 pp claim under max_new=300 was an asymmetric truncation artefact.")
