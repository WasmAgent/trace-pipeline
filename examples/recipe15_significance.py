"""recipe15_significance.py — prove C > A is statistically significant.

Uses eval_trust McNemar + paired bootstrap via evomerge.eval.stat_bridge to
test whether the fine-tuned model (group C) outperforms the base model
(group A) with p < 0.05.

Run:
    python examples/recipe15_significance.py
"""
from evomerge.eval import EvalRecord, paired_significance, compare_all_groups

# ---- synthetic results mirroring recipe14 stubs ----
TASK_IDS = [f"task-{i:03d}" for i in range(30)]

def _records(group: str, pass_fn) -> list[EvalRecord]:
    return [
        EvalRecord(task_id=tid, group=group, final_pass=pass_fn(i))
        for i, tid in enumerate(TASK_IDS)
    ]

group_A = _records("A", lambda i: i % 10 >= 7)   # 30% pass rate
group_B = _records("B", lambda i: i % 10 >= 5)   # 50% pass rate
group_C = _records("C", lambda i: i % 10 >= 2)   # 80% pass rate
group_D = _records("D", lambda i: i % 20 != 0)   # 95% pass rate

# ---- A vs C paired significance ----
report = paired_significance(group_A, group_C, label_a="A (base)", label_b="C (fine-tuned)")

print("=== A vs C Significance Report ===")
print(f"  n_common          : {report.n_common}")
print(f"  pass_rate A       : {report.pass_rate_a:.1%}  95% CI {report.pass_ci_a[0]:.1%}–{report.pass_ci_a[1]:.1%}")
print(f"  pass_rate C       : {report.pass_rate_b:.1%}  95% CI {report.pass_ci_b[0]:.1%}–{report.pass_ci_b[1]:.1%}")
print(f"  delta             : +{report.pass_rate_delta:.1%}")
print(f"  McNemar b/c       : {report.mcnemar_b}/{report.mcnemar_c}")
print(f"  McNemar p         : {report.mcnemar_p:.4f}")
print(f"  significant@0.05  : {report.significant_at_05}")
print(f"  significant@0.01  : {report.significant_at_01}")
print(f"  bootstrap delta   : {report.bootstrap['delta_acc']:.3f}  "
      f"95% CI [{report.bootstrap['ci_lo']:.3f}, {report.bootstrap['ci_hi']:.3f}]")

assert report.significant_at_05, "expected C > A to be significant at p<0.05"

# ---- compare all groups against A ----
print("\n=== All groups vs A ===")
all_groups = {"A": group_A, "B": group_B, "C": group_C, "D": group_D}
comparisons = compare_all_groups(all_groups, reference="A")
for key, r in comparisons.items():
    sig = "✓ p<0.05" if r.significant_at_05 else "  n.s."
    print(f"  {key:<8}  delta={r.pass_rate_delta:+.1%}  p={r.mcnemar_p:.4f}  {sig}")

print("\nall significance checks passed")
