# Examples

> Copy-pasteable recipes for common audit tasks. Every example uses the
> raw data shipped in `data/`, so each block is independently runnable.

## Recipe 1: paired McNemar (the headline test)

You have two greedy-eval JSON logs from candidates A and B, run on the
same item set. Question: is A's accuracy advantage real?

```python
import json
from scipy.stats import binomtest

with open("data/winner_max_new768.json") as f:
    a = json.load(f)
with open("data/instruct_max_new768.json") as f:
    b = json.load(f)

am = {r["id"]: r["correct"] for r in a["results"] if "correct" in r}
bm = {r["id"]: r["correct"] for r in b["results"] if "correct" in r}
common = sorted(set(am) & set(bm))

# Discordant cells: McNemar's test summands
b_count = sum(1 for i in common if bm[i] and not am[i])  # B-only correct
c_count = sum(1 for i in common if not bm[i] and am[i])  # A-only correct

p = binomtest(min(b_count, c_count), b_count + c_count,
              p=0.5, alternative="two-sided").pvalue

print(f"n={len(common)}  b={b_count}  c={c_count}  p={p:.4f}")
# n=199  b=29  c=27  p=0.8939
```

**Interpretation:** With $b=29$, $c=27$, the discordant counts are
nearly equal — there is no significant difference between A and B.
If the same test on the **broken** protocol (max_new=300) gave
$p=0.0151$, the +10 pp improvement was the protocol asymmetry talking,
not the merge.

---

## Recipe 2: Wilson 95% CI per candidate

```python
from eval_trust.paired_stats import wilson_ci

n = 199
n_winner = sum(1 for i in common if am[i])    # 135
n_baseline = sum(1 for i in common if bm[i])  # 137

w_lo, w_hi = wilson_ci(n_winner, n)
b_lo, b_hi = wilson_ci(n_baseline, n)

print(f"winner   {n_winner}/{n} = {n_winner/n*100:.1f}%  "
      f"95% CI [{w_lo*100:.1f}, {w_hi*100:.1f}]")
print(f"baseline {n_baseline}/{n} = {n_baseline/n*100:.1f}%  "
      f"95% CI [{b_lo*100:.1f}, {b_hi*100:.1f}]")

# winner   135/199 = 67.8%  95% CI [61.1, 73.9]
# baseline 137/199 = 68.8%  95% CI [62.1, 74.9]
```

The two CIs overlap heavily — consistent with the McNemar non-significance.

---

## Recipe 3: paired bootstrap CI on the delta

```python
from eval_trust.paired_stats import paired_bootstrap

a_arr = [am[i] for i in common]
b_arr = [bm[i] for i in common]
out = paired_bootstrap(a_arr, b_arr, n_iter=10000, seed=42)

print(f"delta = {out['delta_acc']*100:+.2f} pp")
print(f"95% CI: [{out['ci_lo']*100:+.2f}, {out['ci_hi']*100:+.2f}] pp")
print(f"two-sided p = {out['p_two_sided']:.3f}")
# delta = -1.01 pp
# 95% CI: [-8.04, +6.03] pp   ← contains zero
# two-sided p ≈ 0.79
```

The CI containing zero is the same finding as McNemar's $p > 0.05$,
expressed differently.

---

## Recipe 4: lottery rate (single-greedy noise floor)

You run greedy on $n$ items, get $w$ wrong items. Question: are those
$w$ items *truly* wrong, or is your greedy decode just unlucky?

```python
import json

# Pre-computed on the case-study model: SC-5 majority on the 65
# greedy-wrong items. (See data/self_consistency_full.json structure.)
with open("data/self_consistency_full.json") as f:
    sc = json.load(f)

print(f"Tested:    {sc['n_tested']} greedy-wrong items")
print(f"Recovered: {sc['lottery_count']} ({sc['lottery_rate']*100:.1f}%)")
# Tested:    65 greedy-wrong items
# Recovered: 25 (38.5%)
```

**Interpretation:** 38.5% of greedy-wrong items are recoverable under
SC-5 majority. A delta of less than this lottery rate (~12.5 pp on
absolute accuracy for *single*-model SC-vs-greedy) should be
cross-checked under SC before being claimed.

---

## Recipe 5: T0v2 channel triage (no re-evaluation)

Given an existing greedy-eval JSON with `gen_text`, classify each
wrong answer into one of six channels.

```python
# As a Python module
from eval_trust.t0v2.truncation_extract import classify
from pathlib import Path

out = classify(Path("data/winner_max_new768.json"))
print(f"Wrong items: {out['n_wrong']}")
print(f"Truncated:   {out['n_truncated']} ({out['truncated_share_of_wrong']*100:.1f}%)")
```

Or via the CLI:

```bash
python -m eval_trust.t0v2.truncation_extract data/winner_max_new768.json
```

---

## Recipe 6: aggregate to a routing decision

```python
from eval_trust.t0v2.aggregate import aggregate

verdict = aggregate(
    n_total=200,
    n_wrong=65,
    counts={
        "A_truncated": 17,
        "A_extract_v2": 1,
        "B_stepwise": 4,
        "C_token": 8,
        "Class2": 35,
    },
    lottery_rate=0.385,  # from Recipe 4
)
print(verdict["decision_v2"])
# {'case': 'beta',
#  'action': 'switch to SC-k majority before claiming (lottery=38%)',
#  'alpha_threshold': 0.15,
#  'beta_threshold': 0.30}
```

**Decision logic** (paper §4.3):

| Verdict | Trigger | Action |
|---|---|---|
| **β** | lottery rate ≥ 30 % | switch to SC-$k$ majority *before* claiming any delta |
| **α** | first-class rate ≥ 15 % of total | surface artefacts present; cheap repair available |
| **γ** | otherwise | reasoning bottleneck; report cap and stop |

In the case study: lottery is 38.5 % → **β**. Even though the
first-class rate hits 15 % (matching α), the lottery floor takes
priority — fix sampling first.

---

## Recipe 7: end-to-end (one command)

For a quick check of any pair of greedy-eval JSON files:

```bash
python run_audit.py \
    --winner data/winner_max_new768.json \
    --baseline data/instruct_max_new768.json \
    --sc data/self_consistency_full.json \
    --t0v2 data/t0v2_aggregate.json \
    --json full_report.json
```

Prints the full audit report and dumps a structured JSON.

The defaults point to the case-study data, so just `python run_audit.py`
also works.

---

## Recipe 8: conformal interval for "small-n" accuracy

When $n < 100$, the Wilson interval can be wide and the normal
approximation breaks down. Conformal prediction gives a
distribution-free interval given a calibration set:

```python
import numpy as np
from eval_trust.conformal_ci import compute_conformal_interval

# Calibration: 50 holdout items from a related benchmark, residuals
# (predicted - true accuracy) on each
calibration_residuals = np.random.default_rng(seed=0).normal(0, 0.05, 50)
calibration_residuals = np.abs(calibration_residuals)

# New point estimate: my model's accuracy on a 30-item dev set
my_estimate = 0.625

ci = compute_conformal_interval(
    calibration_residuals,
    prediction=my_estimate,
    alpha=0.10,  # 90 % coverage
)
print(f"Estimate: {my_estimate*100:.1f}%")
print(f"90% conformal interval: [{ci.lower*100:.1f}, {ci.upper*100:.1f}]")
print(f"Width: {ci.width*100:.1f} pp")
```

**When to prefer conformal over Wilson:**

- Wilson assumes the per-item outcomes are i.i.d. binomial.
- Conformal assumes only that the *test* item is exchangeable with the
  calibration set. Useful when residuals are heteroscedastic, when
  the calibration set has a richer structure than just "did each item
  pass", or when you're predicting a continuous quantity.

For straightforward accuracy on a single benchmark, Wilson is fine.
For "the merged model's accuracy on a new domain" given calibration
on related domains, conformal is more honest.

---

## Recipe 9: sample-size planning

How many items do you need to detect a 4.5 pp delta with 80 % power
at α = 0.05?

```python
from eval_trust.paired_stats import n_for_power

n_needed = n_for_power(delta=0.045, power=0.80, alpha=0.05)
print(f"Need n >= {n_needed} paired items")
# Need n >= 952 paired items
```

For 10 pp:

```python
print(f"For 10 pp: n >= {n_for_power(0.10):.0f}")
# For 10 pp: n >= 193
```

The case study's 200-item dev set was right-sized for a 10 pp delta —
which is why the +10 pp claim *did* reach paired significance under
the broken protocol. The protocol was the bug, not the n.

---

## Where to go next

- The paper, [`papers/eval_trust/draft.pdf`](papers/eval_trust/draft.pdf),
  walks through the case study in full and gives the formal
  channel definitions in Appendix C.
- The audit checklist in Appendix A is the version designed for use
  on a paper draft you're about to submit.
- `numbers_cross_check.json` has every number cited in the paper with
  a source-file pointer, in case you want to recompute anything.
