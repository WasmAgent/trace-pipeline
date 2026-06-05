"""eval_trust.t0v2.aggregate — aggregate T0v2 channel verdicts.

Reads per-channel reports (e.g. truncation_extract.json) and produces a
single routing decision per the paper section 4.3:

  - alpha: first-class rate >= 15% -> surgical repair worth pursuing
  - beta:  lottery rate    >= 30% -> switch to SC majority before claiming
  - gamma: otherwise              -> reasoning bottleneck, accept the cap

Channels (paper section 4.2):
    A_truncated, A_extract_v2, B_stepwise, B_selfcorrect_regress,
    C_token, Class2.
"""
from __future__ import annotations

import argparse
import json
import sys

CHANNELS = (
    "A_truncated", "A_extract_v2", "B_stepwise",
    "B_selfcorrect_regress", "C_token", "Class2",
)
FIRST_CLASS = (
    "A_truncated", "A_extract_v2", "B_stepwise",
    "B_selfcorrect_regress", "C_token",
)


def aggregate(
    n_total: int,
    n_wrong: int,
    counts: dict,
    lottery_rate: float | None = None,
    alpha_threshold: float = 0.15,
    beta_threshold: float = 0.30,
) -> dict:
    """Aggregate per-channel counts into a routing decision."""
    n_first_class = sum(counts.get(ch, 0) for ch in FIRST_CLASS)
    first_class_rate_of_total = n_first_class / n_total if n_total else 0.0

    if lottery_rate is not None and lottery_rate >= beta_threshold:
        decision = "beta"
        action = (
            f"switch to SC-k majority before claiming "
            f"(lottery={lottery_rate*100:.0f}%)"
        )
    elif first_class_rate_of_total >= alpha_threshold:
        decision = "alpha"
        action = (
            f"first-class rate {first_class_rate_of_total*100:.1f}% meets "
            f"threshold; consider repair"
        )
    else:
        decision = "gamma"
        action = "reasoning bottleneck dominates; accept cap or change approach"

    return {
        "n_total": n_total,
        "n_wrong": n_wrong,
        "n_first_class": n_first_class,
        "first_class_rate_of_total": first_class_rate_of_total,
        "first_class_rate_of_wrong": (
            n_first_class / n_wrong if n_wrong else 0.0
        ),
        "verdict_counts": {k: counts.get(k, 0) for k in CHANNELS},
        "lottery_rate": lottery_rate,
        "decision_v2": {
            "case": decision,
            "action": action,
            "alpha_threshold": alpha_threshold,
            "beta_threshold": beta_threshold,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--wrong", type=int, required=True)
    for ch in CHANNELS:
        ap.add_argument(f"--{ch}", type=int, default=0)
    ap.add_argument("--lottery_rate", type=float, default=None)
    args = ap.parse_args()

    counts = {ch: getattr(args, ch) for ch in CHANNELS}
    out = aggregate(args.total, args.wrong, counts, lottery_rate=args.lottery_rate)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
