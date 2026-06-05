"""eval_trust.conformal_ci — distribution-free confidence intervals.

Split conformal prediction: take quantiles of calibration-set residuals to
construct distribution-free prediction intervals.

Math:
    On calibration set {(x_i, y_i)} compute residuals r_i = |y_i - y_pred_i|.
    Take q = quantile(r_1..r_n, ceil((n+1)*alpha)/n).
    Interval for new prediction y_pred*: [y_pred* - q, y_pred* + q].

Theoretical guarantee (Vovk et al.):
    P(y* in [y_pred* - q, y_pred* + q]) >= 1 - alpha
    Assumes only i.i.d.; no assumption on residual distribution shape.

Reference:
    Vovk et al. 2005, Algorithmic Learning in a Random World.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalInterval:
    """Conformal prediction interval result."""

    lower: float
    upper: float
    width: float
    quantile: float


def compute_conformal_interval(
    calibration_errors: list[float] | np.ndarray,
    prediction: float,
    *,
    alpha: float = 0.10,
) -> ConformalInterval:
    """Compute distribution-free prediction interval using conformal prediction.

    Args:
        calibration_errors: |y_true - y_pred| for calibration set.
        prediction: Point prediction for new sample.
        alpha: Miscoverage rate (0.10 = 90% coverage).

    Returns:
        ConformalInterval with lower, upper bounds.
    """
    errors = np.asarray(calibration_errors, dtype=np.float64)
    n = len(errors)
    if n < 2:
        return ConformalInterval(
            lower=prediction,
            upper=prediction,
            width=0.0,
            quantile=0.0,
        )

    # Conformal quantile: ceil((n+1)(1-alpha))/n
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = float(np.quantile(errors, level))
    return ConformalInterval(
        lower=prediction - q,
        upper=prediction + q,
        width=2 * q,
        quantile=q,
    )


def calibrate_from_holdout(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    *,
    alpha: float = 0.10,
) -> callable:
    """Create a calibrated interval predictor from holdout data.

    Returns a function: prediction → ConformalInterval.
    """
    errors = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))

    def predict_interval(prediction: float) -> ConformalInterval:
        return compute_conformal_interval(errors, prediction, alpha=alpha)

    return predict_interval


__all__ = [
    "ConformalInterval",
    "calibrate_from_holdout",
    "compute_conformal_interval",
]