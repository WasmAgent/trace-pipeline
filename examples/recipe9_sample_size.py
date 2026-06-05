"""examples/recipe9_sample_size.py — sample-size planning.

Run from repo root:
    python examples/recipe9_sample_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.paired_stats import n_for_power

print("Sample size needed to detect an absolute delta with 80% power, alpha=0.05:\n")
for delta_pp in [2.5, 4.5, 7.5, 10.0, 15.0]:
    n_needed = n_for_power(delta=delta_pp / 100, power=0.80, alpha=0.05)
    print(f"  delta = {delta_pp:5.1f} pp  ->  n >= {n_needed}")

print()
print("The case study used n=200 (enough for ~10pp delta, exactly what")
print("Phase 13 thought it had). The protocol was the bug, not the n.")
