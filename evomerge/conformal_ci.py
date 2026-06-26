"""
conformal_ci.py — Wilson confidence interval computation for agent benchmark reports.

No external dependencies: pure Python math only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCI:
    """Wilson CI result for a single model benchmark run."""

    model_id: str
    pass_count: int
    n_trials: int
    pass_rate: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g. 0.95


# ---------------------------------------------------------------------------
# Core math helper
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float) -> tuple[float, float]:
    """Compute Wilson score confidence interval.

    Parameters
    ----------
    k:
        Number of successes.
    n:
        Total number of trials.
    z:
        z-score for the desired confidence level
        (e.g. 1.96 for 95 %, 2.576 for 99 %).

    Returns
    -------
    (lower, upper) — both in [0, 1].
    """
    if n == 0:
        return 0.0, 1.0

    p_hat = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )

    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return lower, upper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_benchmark_ci(
    model_id: str,
    pass_count: int,
    n_trials: int,
    ci_level: float = 0.95,
) -> BenchmarkCI:
    """Compute a Wilson CI for one model.

    Parameters
    ----------
    model_id:
        Identifier string for the model being evaluated.
    pass_count:
        Number of passing (successful) trials.
    n_trials:
        Total number of trials.
    ci_level:
        Confidence level as a fraction in (0, 1), e.g. 0.95.

    Returns
    -------
    BenchmarkCI dataclass.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_trials < 0:
        raise ValueError(f"n_trials must be >= 0, got {n_trials}")
    if not 0 <= pass_count <= n_trials:
        raise ValueError(
            f"pass_count ({pass_count}) must be between 0 and n_trials ({n_trials})"
        )

    # Two-sided z-score: solve Φ(z) = 1 - (1 - ci_level) / 2
    # Using the Beasley-Springer-Moro rational approximation for the inverse normal CDF.
    alpha = 1.0 - ci_level
    p_tail = 1.0 - alpha / 2.0
    z = _inverse_normal_cdf(p_tail)

    pass_rate = pass_count / n_trials if n_trials > 0 else 0.0
    lower, upper = _wilson_ci(pass_count, n_trials, z)

    return BenchmarkCI(
        model_id=model_id,
        pass_count=pass_count,
        n_trials=n_trials,
        pass_rate=pass_rate,
        ci_lower=lower,
        ci_upper=upper,
        ci_level=ci_level,
    )


def benchmark_ci_report(
    pass_counts_dict: dict[str, int],
    n_trials: int,
    ci_level: float = 0.95,
) -> str:
    """Compute Wilson CIs for multiple models and return a Markdown table.

    Parameters
    ----------
    pass_counts_dict:
        Mapping of model_id → pass_count.
    n_trials:
        Total number of trials (shared across all models).
    ci_level:
        Confidence level, e.g. 0.95.

    Returns
    -------
    A Markdown table string with columns:
    Model | Pass | N | Pass Rate | CI Lower | CI Upper | CI Level
    """
    rows: list[BenchmarkCI] = []
    for model_id, pass_count in pass_counts_dict.items():
        rows.append(compute_benchmark_ci(model_id, pass_count, n_trials, ci_level))

    pct = int(round(ci_level * 100))
    header = (
        f"| Model | Pass | N | Pass Rate | CI Lower ({pct}%) | CI Upper ({pct}%) |\n"
        f"| ----- | ---: | --: | --------: | ----------------: | ----------------: |"
    )

    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.model_id} "
            f"| {r.pass_count} "
            f"| {r.n_trials} "
            f"| {r.pass_rate:.3f} "
            f"| {r.ci_lower:.3f} "
            f"| {r.ci_upper:.3f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inverse normal CDF — rational approximation (Abramowitz & Stegun 26.2.17)
# Accurate to ~5 significant figures; no scipy dependency required.
# ---------------------------------------------------------------------------

def _inverse_normal_cdf(p: float) -> float:
    """Return z such that Φ(z) ≈ p, for 0 < p < 1."""
    # Coefficients for rational approximation
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    sign = 1.0
    q = p
    if q > 0.5:
        q = 1.0 - q
        sign = -1.0

    t = math.sqrt(-2.0 * math.log(q))
    numerator = c[0] + c[1] * t + c[2] * t * t
    denominator = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    z = t - numerator / denominator

    return sign * z
