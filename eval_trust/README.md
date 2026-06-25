# eval_trust — Benchmark Audit Toolkit

**A lightweight, dependency-free referee for LLM benchmark claims.**

`eval_trust` exists because benchmark numbers can look significant while
being entirely explained by experimental artifacts — truncated generations,
contaminated test sets, or insufficient sample sizes. This toolkit gives you
the statistical machinery to check a claim before acting on it.

---

## The case that inspired this

We spent **5 months** validating a +10 pp GSM8K improvement on a Qwen2.5-1.5B
merge: three rounds of manual review, paired McNemar (p = 0.015), multi-seed
runs. Then we noticed `max_new_tokens=300` in the generation script.

Re-ran with `max_new_tokens=768`:

| | Before audit (max\_new=300) | After audit (max\_new=768) |
|---|---|---|
| Delta | +10.0 pp | −1.0 pp |
| McNemar b/c | 21 / 41 | 29 / 27 |
| p-value | 0.015 | **0.894** |
| Conclusion | "Significant improvement" | Not significant |

The "improvement" was the baseline being truncated more often than the
candidate — an asymmetric artifact that mimics real learning signal.

Full story: [`papers/eval_trust/draft.pdf`](../papers/eval_trust/)

---

## What it checks

| Check | Function / Command |
|---|---|
| Is the paired delta real? | `mcnemar_exact`, `paired_bootstrap` |
| What's the accuracy confidence interval? | `wilson_ci` |
| Were generations truncated? | `truncation_extract`, `evomerge truncation` |
| Does the result replicate across seeds? | `paired_bootstrap` with different seeds |
| Is the test set contaminated? | `evomerge validate --contamination-threshold` |
| What's the minimum sample size needed? | `n_for_power` |
| Does the benchmark have exploit surfaces? | `ExploitSurface`, `evomerge lint-benchmark` |

---

## Install

```bash
# As part of trace-pipeline
pip install evomerge         # includes eval_trust

# Or use directly from source (zero extra deps beyond numpy)
git clone https://github.com/WasmAgent/trace-pipeline
cd trace-pipeline
pip install -e "."
```

No GPU. No model inference. Pure Python + NumPy (+ SciPy for `binomtest`).

---

## 30-second quickstart — reproduce the case study

```python
from scipy.stats import binomtest

# ORIGINAL run (max_new_tokens=300) — claimed significant
b, c = 21, 41   # b = B-only correct, c = A-only correct
p_before = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue
print(f"Before audit: p = {p_before:.4f}")   # → 0.0151  ← looked significant

# AUDITED run (max_new_tokens=768) — the real answer
b, c = 29, 27
p_after = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue
print(f"After  audit: p = {p_after:.4f}")    # → 0.8940  ← not significant
```

Or run the bundled demo:

```bash
python examples/recipe1_paired_mcnemar.py
```

---

## Core API

### Paired McNemar test

```python
from eval_trust.paired_stats import mcnemar_exact

# a_correct[i] = True if model A got item i right (same item set for both)
p = mcnemar_exact(b=29, c=27)   # two-sided exact test
print(f"p = {p:.4f}")           # → 0.8940
```

### Wilson confidence interval

```python
from eval_trust.paired_stats import wilson_ci

lo, hi = wilson_ci(correct=148, n=199, conf=0.95)
print(f"95% CI: [{lo:.3f}, {hi:.3f}]")
# → 95% CI: [0.676, 0.810]
```

### Paired bootstrap

```python
from eval_trust.paired_stats import paired_bootstrap

# Returns (delta_accuracy, (ci_lo, ci_hi))
delta, ci = paired_bootstrap(correct_a=[True, False, ...],
                             correct_b=[False, True, ...],
                             n_iter=10000)
print(f"delta = {delta:+.3f}  95% CI: {ci}")
```

### Sample size calculator

```python
from eval_trust.paired_stats import n_for_power

# How many items do you need to detect a 5 pp delta with 80% power?
n = n_for_power(delta=0.05, power=0.80, alpha=0.05)
print(f"Required n ≥ {n}")   # → 783
```

### Truncation detection

```python
from eval_trust.t0v2.truncation_extract import extract_truncation_stats

stats = extract_truncation_stats("path/to/lm_eval_samples.jsonl")
print(f"Truncation rate: {stats.truncation_rate:.1%}")
print(f"Avg tokens used: {stats.mean_tokens:.0f} / {stats.max_new_tokens}")
```

### Exploit surface linter

```python
from eval_trust.exploit_surface import ExploitSurface

findings = ExploitSurface.from_task_dir("path/to/benchmark/tasks/")
for f in findings:
    if f.severity in ("high", "critical"):
        print(f"[{f.severity}] {f.category}: {f.description}")
```

---

## CLI (via evomerge)

`eval_trust` is also accessible through the `evomerge` CLI:

```bash
# Full audit in one shot
python -m eval_trust audit \
  --result-a data/winner_max_new768.json \
  --result-b data/instruct_max_new768.json

# Paired significance only
python -m eval_trust paired \
  --result-a results_a.jsonl \
  --result-b results_b.jsonl

# Truncation detection
python -m eval_trust truncation \
  --samples lm_eval_samples.jsonl \
  --max-new-tokens 300

# Contamination check (evomerge)
python -m evomerge validate \
  --rollout rollouts.jsonl \
  --contamination-threshold 0.05
```

---

## Recipes

17 copy-pasteable examples in [`examples/`](../examples/):

| Recipe | Task |
|---|---|
| `recipe1_paired_mcnemar.py` | Reproduce the case-study flip |
| `recipe2_wilson_ci.py` | Confidence interval for a single accuracy |
| `recipe3_paired_bootstrap.py` | Bootstrap CI on a paired delta |
| `recipe4_lottery_rate.py` | Estimate true pass@k from a sample |
| `recipe5_t0v2_truncation.py` | Detect truncated T0v2 generations |
| `recipe6_aggregate_decision.py` | T0v2 aggregate audit decision |
| `recipe7_end_to_end.py` | Full eval_trust pipeline in one script |
| `recipe8_conformal.py` | Split-conformal accuracy interval |
| `recipe9_sample_size.py` | Required n for a given power target |
| `recipe10_lm_eval_bridge.py` | Bridge to lm-evaluation-harness logs |
| `recipe15_significance.py` | Significance for an A/B/C eval harness |

---

## What eval_trust is not

- **Not a benchmark runner** — use `lm-evaluation-harness`, `HELM`, `inspect-ai`, etc.
- **Not a leaderboard** — it's the referee, not the scoreboard.
- **Not specific to WasmAgent** — any two result files with per-item correctness
  arrays can be audited. The WasmAgent connection is that `evomerge` uses
  `eval_trust` to gate traces before they enter training data.

---

## Connection to the WasmAgent data loop

```
agent run  →  bscode export  →  evomerge validate   →  training data
                                    │
                                eval_trust checks:
                                  ✓ no contamination
                                  ✓ sufficient n for any claimed delta
                                  ✓ no truncation artifacts
                                  ✓ no exploit surfaces in the benchmark
```

Traces that pass `eval_trust`'s gates get a `benchmark_trust` score > 0.9 in
`AgentTrustScore`. Traces that fail are quarantined and never enter DPO/PPO
training.

→ [ENTERPRISE_AUDIT_DEMO.md](../docs/ENTERPRISE_AUDIT_DEMO.md) for the full audit pipeline
→ [TRACE_TO_TRAINING_10MIN.md](../docs/TRACE_TO_TRAINING_10MIN.md) for the training data tutorial

---

## Paper

📄 *"Silent Contamination in LLM Merging Evaluation"*
[`papers/eval_trust/draft.pdf`](../papers/eval_trust/)

Covers: truncation artifacts, McNemar significance collapse, multi-seed
reproducibility, and a case study on Qwen2.5-1.5B merge evaluation.
