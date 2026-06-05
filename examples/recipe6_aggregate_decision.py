"""examples/recipe6_aggregate_decision.py — alpha/beta/gamma routing.

Run from repo root:
    python examples/recipe6_aggregate_decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.t0v2.aggregate import aggregate

verdict = aggregate(
    n_total=200,
    n_wrong=65,
    counts={
        "A_truncated": 17,
        "A_extract_v2": 1,
        "B_stepwise": 4,
        "C_token": 8,
        "Class2": 35,
    },
    lottery_rate=0.385,
)

print(json.dumps(verdict["decision_v2"], indent=2))
print()
print("In the case study: lottery 38.5% triggers BETA verdict.")
print("Even though the first-class rate hits 15% (would be ALPHA), the")
print("lottery floor takes priority - fix sampling protocol first.")
