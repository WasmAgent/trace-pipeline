# Examples — Runnable Recipes

> Each file in this directory is a standalone recipe from
> [`EXAMPLES.md`](../EXAMPLES.md), runnable directly without copy-paste.

```bash
# From repo root:
python examples/recipe1_paired_mcnemar.py
python examples/recipe2_wilson_ci.py
python examples/recipe3_paired_bootstrap.py
python examples/recipe4_lottery_rate.py
python examples/recipe5_t0v2_truncation.py
python examples/recipe6_aggregate_decision.py
python examples/recipe7_end_to_end.py
python examples/recipe8_conformal.py
python examples/recipe9_sample_size.py
python examples/recipe10_lm_eval_bridge.py
```

Each recipe:

| # | File | What it shows |
|---|---|---|
| 1 | `recipe1_paired_mcnemar.py` | The headline test on the case-study data |
| 2 | `recipe2_wilson_ci.py` | Wilson 95% CI per candidate |
| 3 | `recipe3_paired_bootstrap.py` | Paired bootstrap CI on the delta |
| 4 | `recipe4_lottery_rate.py` | SC-5 lottery rate (single-greedy noise floor) |
| 5 | `recipe5_t0v2_truncation.py` | A_truncated channel detector |
| 6 | `recipe6_aggregate_decision.py` | alpha/beta/gamma routing decision |
| 7 | `recipe7_end_to_end.py` | The full case-study reproducer (`run_audit.py`) |
| 8 | `recipe8_conformal.py` | Conformal interval for small-n accuracy |
| 9 | `recipe9_sample_size.py` | Sample-size planning |
| 10 | `recipe10_lm_eval_bridge.py` | Convert lm-evaluation-harness output |

For prose explanations and theory references, see
[`EXAMPLES.md`](../EXAMPLES.md). For a single-command run of all
recipes (with assertions on outputs), see
`benchmarks/run_all_examples.py` (TODO).
