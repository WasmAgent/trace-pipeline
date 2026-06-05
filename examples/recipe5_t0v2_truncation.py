"""examples/recipe5_t0v2_truncation.py — A_truncated channel detector.

Run from repo root:
    python examples/recipe5_t0v2_truncation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.t0v2.truncation_extract import classify

DATA = Path(__file__).resolve().parent.parent / "data"

out = classify(DATA / "winner_max_new768.json")

print(f"Wrong items: {out['n_wrong']}")
print(f"Truncated:   {out['n_truncated']}  "
      f"({out['truncated_share_of_wrong']*100:.1f}% of wrong)")
print()
print(f"First 10 truncated IDs: {out['truncated_ids'][:10]}")
