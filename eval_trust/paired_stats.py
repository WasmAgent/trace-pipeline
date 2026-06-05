"""eval_trust.paired_stats — paired statistics for benchmark deltas.

Provides three classical tools for evaluating whether a small benchmark delta
is real or noise, configured for the use case of comparing two LLM candidates
on the same item set:

  - wilson_ci(correct, n, conf=0.95) -> (lo, hi)
        Wilson score interval for a single configuration's accuracy.
  - mcnemar_exact(b, c) -> exact two-sided p-value
        Where b = items A-correct/B-wrong, c = A-wrong/B-correct.
        The standard test for paired binary data.
  - paired_bootstrap(correct_a, correct_b, n_iter=10000) -> (delta_acc, 95% CI)
        Paired-resample bootstrap on the per-item correctness arrays.

Example:

  >>> from eval_trust.paired_stats import mcnemar_exact, wilson_ci
  >>> # On the audited Qwen 1.5B run (n=199, paper sec 3.4):
  >>> #   b=29 (Instruct correct, winner wrong)
  >>> #   c=27 (Instruct wrong, winner correct)
  >>> p = mcnemar_exact(b=29, c=27)
  >>> round(p, 3)
  0.894
  >>> # Compare to the pre-audit (max_new=300) run:
  >>> p_before = mcnemar_exact(b=21, c=41)
  >>> round(p_before, 4)
  0.0151

References:
  Wilson 1927 (Wilson interval), McNemar 1947 (paired binary test),
  Efron 1979 (bootstrap).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson_ci(correct: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion.

    More robust than the normal approximation at small n or extreme p.
    """
    if n == 0:
        return (0.0, 1.0)
    if conf <= 0 or conf >= 1:
        raise ValueError(f"conf must be in (0, 1); got {conf}")

    from statistics import NormalDist

    z = NormalDist(0, 1).inv_cdf(1 - (1 - conf) / 2)
    p_hat = correct / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_coeff(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes.

    Under H0, b ~ Binomial(b+c, 0.5). Returns two-sided exact p in [0, 1].
    """
    n = b + c
    if n == 0:
        return 1.0
    k_min = min(b, c)
    cum = sum(_binom_coeff(n, k) for k in range(k_min + 1))
    one_sided_p = cum / (2**n)
    return min(1.0, 2 * one_sided_p)


def paired_bootstrap(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_iter: int = 10000,
    conf: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of (acc_A - acc_B) sampling distribution.

    Returns dict with delta_acc, ci_lo, ci_hi, p_two_sided, n_iter, n_problems.
    """
    import random

    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must be the same length (paired design)")
    n = len(correct_a)
    if n == 0:
        return {"delta_acc": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_two_sided": 1.0}

    rng = random.Random(seed)
    a_arr = [bool(x) for x in correct_a]
    b_arr = [bool(x) for x in correct_b]
    point = sum(a_arr) / n - sum(b_arr) / n

    deltas = []
    for _ in range(n_iter):
        idxs = [rng.randrange(n) for _ in range(n)]
        acc_a = sum(a_arr[i] for i in idxs) / n
        acc_b = sum(b_arr[i] for i in idxs) / n
        deltas.append(acc_a - acc_b)

    deltas.sort()
    alpha = (1 - conf) / 2
    lo_idx = max(0, int(alpha * n_iter))
    hi_idx = min(n_iter - 1, int((1 - alpha) * n_iter))

    n_extreme = sum(1 for d in deltas if (point > 0 and d <= 0) or (point < 0 and d >= 0))
    p_two = min(1.0, 2 * (n_extreme + 1) / (n_iter + 1))

    return {
        "delta_acc": point,
        "ci_lo": deltas[lo_idx],
        "ci_hi": deltas[hi_idx],
        "p_two_sided": p_two,
        "n_iter": n_iter,
        "n_problems": n,
    }


def n_for_power(delta: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Approximate sample size to detect an absolute delta (paired binary)."""
    from statistics import NormalDist

    z_alpha = NormalDist(0, 1).inv_cdf(1 - alpha / 2)
    z_beta = NormalDist(0, 1).inv_cdf(power)
    sigma = 0.5  # worst case for binary diff
    return int(math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2))


__all__ = ["mcnemar_exact", "n_for_power", "paired_bootstrap", "wilson_ci"]
