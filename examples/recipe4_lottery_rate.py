"""examples/recipe4_lottery_rate.py — single-greedy noise floor.

Run from repo root:
    python examples/recipe4_lottery_rate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data"

with open(DATA / "self_consistency_full.json") as f:
    sc = json.load(f)

print(f"Tested:    {sc['n_tested']} greedy-wrong items")
print(f"Recovered: {sc['lottery_count']} ({sc['lottery_rate']*100:.1f}%)")
print()
print("38.5% of greedy-wrong items are recoverable under SC-5 majority.")
print("Single-greedy has a ~12.5pp 'noise floor' on this configuration.")
print("Any reported delta below this should be cross-checked under SC.")
