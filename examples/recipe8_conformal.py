"""examples/recipe8_conformal.py — conformal CI for small-n accuracy.

Run from repo root:
    python examples/recipe8_conformal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval_trust.conformal_ci import compute_conformal_interval

# Calibration: 50 holdout items from a related benchmark; treat the
# residuals (predicted - true accuracy) as a sample of typical errors.
calibration_residuals = np.abs(
    np.random.default_rng(seed=0).normal(0, 0.05, 50)
)

# New point estimate: my model's accuracy on a 30-item dev set
my_estimate = 0.625

ci = compute_conformal_interval(
    calibration_residuals,
    prediction=my_estimate,
    alpha=0.10,  # 90% coverage
)

print(f"Estimate: {my_estimate*100:.1f}%")
print(f"90% conformal interval: [{ci.lower*100:.1f}, {ci.upper*100:.1f}]")
print(f"Width: {ci.width*100:.1f} pp")
print()
print("When n < 100, conformal can be more honest than Wilson because it")
print("only assumes calibration items are exchangeable with the test item")
print("(no i.i.d. binomial assumption).")
