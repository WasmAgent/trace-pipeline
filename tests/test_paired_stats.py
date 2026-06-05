"""tests/test_paired_stats.py — paired statistics unit tests."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))


def test_wilson_ci_22_at_13():
    from eval_trust.paired_stats import wilson_ci

    lo, hi = wilson_ci(13, 22, conf=0.95)
    assert 0.38 < lo < 0.40, f"lo = {lo}"
    assert 0.76 < hi < 0.78, f"hi = {hi}"


def test_wilson_ci_22_at_12():
    from eval_trust.paired_stats import wilson_ci

    lo, hi = wilson_ci(12, 22, conf=0.95)
    assert 0.34 < lo < 0.36, f"lo = {lo}"
    assert 0.72 < hi < 0.74, f"hi = {hi}"


def test_wilson_ci_extreme_cases():
    from eval_trust.paired_stats import wilson_ci

    lo, hi = wilson_ci(10, 10)
    assert lo > 0.6
    assert hi == 1.0
    lo, hi = wilson_ci(0, 10)
    assert lo < 1e-15
    assert hi > 0.0
    # n=0：[0, 1]
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_mcnemar_b1_c0_not_significant():
    from eval_trust.paired_stats import mcnemar_exact

    p = mcnemar_exact(b=1, c=0)
    # b=1, c=0 → n=1, k_min=0, cum=C(1,0)=1, p_one=1/2, p_two=1.0
    assert p == 1.0


def test_mcnemar_strong_effect():
    from eval_trust.paired_stats import mcnemar_exact

    p = mcnemar_exact(b=10, c=0)
    # n=10, k_min=0, p_one = 1/1024, p_two ≈ 0.002
    assert p < 0.01


def test_mcnemar_balanced_no_effect():
    from eval_trust.paired_stats import mcnemar_exact

    p = mcnemar_exact(b=5, c=5)
    assert p == 1.0


def test_mcnemar_zero_disagreement():
    from eval_trust.paired_stats import mcnemar_exact

    p = mcnemar_exact(b=0, c=0)
    assert p == 1.0


def test_paired_bootstrap_no_difference():
    from eval_trust.paired_stats import paired_bootstrap

    a = [True, False, True, True, False] * 10
    b = list(a)
    res = paired_bootstrap(a, b, n_iter=1000, seed=0)
    assert abs(res["delta_acc"]) < 1e-9
    assert res["ci_lo"] <= 0 <= res["ci_hi"]


def test_paired_bootstrap_clear_difference():
    from eval_trust.paired_stats import paired_bootstrap

    n = 50
    a = [True] * n
    b = [True] * 30 + [False] * 20
    res = paired_bootstrap(a, b, n_iter=1000, seed=0)
    assert 0.35 < res["delta_acc"] < 0.45  # 0.4


def test_paired_bootstrap_length_mismatch_raises():
    import pytest

    from eval_trust.paired_stats import paired_bootstrap

    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([True, False], [True, False, True])


def test_n_for_power_recommendations():
    from eval_trust.paired_stats import n_for_power

    n_4_5 = n_for_power(0.045, power=0.80)
    assert 700 < n_4_5 < 1500, f"n_4.5pp = {n_4_5}"
    n_10 = n_for_power(0.10, power=0.80)
    assert 150 < n_10 < 300, f"n_10pp = {n_10}"
    n_15 = n_for_power(0.15, power=0.80)
    assert 60 < n_15 < 130, f"n_15pp = {n_15}"