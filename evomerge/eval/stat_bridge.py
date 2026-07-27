"""Bridge between EvalRecord groups and eval_trust paired statistics.

Takes two groups' EvalRecord lists (paired by task_id) and produces a
structured significance report using eval_trust.paired_stats primitives:
  - McNemar exact test on taskspec_pass (final_pass)
  - Paired bootstrap on pass rate delta
  - Wilson CIs for each group
  - Tool-call validity McNemar

Typical usage (proving C > A is statistically significant):

    from evomerge.eval.stat_bridge import paired_significance

    report = paired_significance(
        records_a=group_a_records,
        records_b=group_c_records,
        label_a="A (base)",
        label_b="C (fine-tuned)",
    )
    print(report.to_dict())
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eval_trust.paired_stats import mcnemar_exact, paired_bootstrap, wilson_ci
from evomerge.eval.metrics import EvalRecord


@dataclass
class SignificanceReport:
    """Paired significance test results between two eval groups."""
    label_a: str
    label_b: str
    n_common: int

    # per-group pass rates
    pass_rate_a: float
    pass_rate_b: float
    pass_rate_delta: float

    # Wilson CIs
    pass_ci_a: tuple[float, float]
    pass_ci_b: tuple[float, float]

    # McNemar on pass/fail
    mcnemar_b: int   # A-pass, B-fail
    mcnemar_c: int   # A-fail, B-pass
    mcnemar_p: float

    # Paired bootstrap
    bootstrap: dict = field(default_factory=dict)

    # Optional: tool-call validity McNemar
    tool_mcnemar_p: float | None = None

    @property
    def significant_at_05(self) -> bool:
        return self.mcnemar_p < 0.05

    @property
    def significant_at_01(self) -> bool:
        return self.mcnemar_p < 0.01

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_common": self.n_common,
            "pass_rate_a": round(self.pass_rate_a, 4),
            "pass_rate_b": round(self.pass_rate_b, 4),
            "pass_rate_delta": round(self.pass_rate_delta, 4),
            "pass_ci_a": [round(v, 4) for v in self.pass_ci_a],
            "pass_ci_b": [round(v, 4) for v in self.pass_ci_b],
            "mcnemar_b": self.mcnemar_b,
            "mcnemar_c": self.mcnemar_c,
            "mcnemar_p": round(self.mcnemar_p, 6),
            "significant_at_05": self.significant_at_05,
            "significant_at_01": self.significant_at_01,
            "bootstrap": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.bootstrap.items()
            },
            "tool_mcnemar_p": (
                round(self.tool_mcnemar_p, 6)
                if self.tool_mcnemar_p is not None
                else None
            ),
        }


def paired_significance(
    records_a: list[EvalRecord],
    records_b: list[EvalRecord],
    *,
    label_a: str = "A",
    label_b: str = "B",
    bootstrap_iters: int = 10_000,
) -> SignificanceReport:
    """Compute paired significance tests between two eval groups.

    Records are paired by task_id. Only tasks present in both groups are used.

    Args:
        records_a: EvalRecord list for group A (base / reference).
        records_b: EvalRecord list for group B (improved / candidate).
        label_a: human-readable label for group A.
        label_b: human-readable label for group B.
        bootstrap_iters: number of bootstrap resamples.

    Returns:
        SignificanceReport with McNemar p-value, bootstrap CI, and Wilson CIs.
    """
    map_a = {r.task_id: r for r in records_a}
    map_b = {r.task_id: r for r in records_b}
    common = sorted(set(map_a) & set(map_b))
    n = len(common)

    if n == 0:
        raise ValueError(
            "No common task_ids between the two groups — cannot compute paired statistics."
        )

    pass_a = [map_a[tid].final_pass for tid in common]
    pass_b = [map_b[tid].final_pass for tid in common]

    n_pass_a = sum(pass_a)
    n_pass_b = sum(pass_b)

    # McNemar b/c counts
    b = sum(1 for pa, pb in zip(pass_a, pass_b, strict=False) if pa and not pb)
    c = sum(1 for pa, pb in zip(pass_a, pass_b, strict=False) if not pa and pb)

    boot = paired_bootstrap(pass_a, pass_b, n_iter=bootstrap_iters)

    # Tool-call validity McNemar (only for tasks where both groups made tool calls)
    tool_p: float | None = None
    tc_pairs = [
        (map_a[tid], map_b[tid])
        for tid in common
        if map_a[tid].tool_calls_total > 0 and map_b[tid].tool_calls_total > 0
    ]
    if tc_pairs:
        tc_valid_a = [ra.tool_calls_valid == ra.tool_calls_total for ra, _ in tc_pairs]
        tc_valid_b = [rb.tool_calls_valid == rb.tool_calls_total for _, rb in tc_pairs]
        tc_b = sum(1 for va, vb in zip(tc_valid_a, tc_valid_b, strict=False) if va and not vb)
        tc_c = sum(1 for va, vb in zip(tc_valid_a, tc_valid_b, strict=False) if not va and vb)
        tool_p = mcnemar_exact(tc_b, tc_c)

    return SignificanceReport(
        label_a=label_a,
        label_b=label_b,
        n_common=n,
        pass_rate_a=n_pass_a / n,
        pass_rate_b=n_pass_b / n,
        pass_rate_delta=(n_pass_b - n_pass_a) / n,
        pass_ci_a=wilson_ci(n_pass_a, n),
        pass_ci_b=wilson_ci(n_pass_b, n),
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_p=mcnemar_exact(b, c),
        bootstrap=boot,
        tool_mcnemar_p=tool_p,
    )


def compare_all_groups(
    group_records: dict[str, list[EvalRecord]],
    *,
    reference: str = "A",
    bootstrap_iters: int = 10_000,
) -> dict[str, SignificanceReport]:
    """Compare every group against a reference group.

    Args:
        group_records: dict mapping group label to EvalRecord list.
        reference: the baseline group label to compare against.
        bootstrap_iters: passed to paired_significance.

    Returns:
        Dict mapping "{reference}_vs_{other}" to SignificanceReport.
    """
    if reference not in group_records:
        raise ValueError(f"Reference group {reference!r} not found in group_records")

    ref_records = group_records[reference]
    results: dict[str, SignificanceReport] = {}
    for label, records in group_records.items():
        if label == reference:
            continue
        key = f"{reference}_vs_{label}"
        results[key] = paired_significance(
            ref_records, records,
            label_a=reference, label_b=label,
            bootstrap_iters=bootstrap_iters,
        )
    return results


__all__ = ["SignificanceReport", "compare_all_groups", "paired_significance"]
