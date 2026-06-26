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

---

## Recipe 10: convert lm-eval-harness output for paired audit

Run [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
for the standard metric, then audit on top:

```bash
# Step 1 — run lm-eval-harness yourself (separate install)
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct \
        --tasks gsm8k --output_path lm_out/instruct.json --log_samples
lm_eval --model hf --model_args pretrained=path/to/your-merge \
        --tasks gsm8k --output_path lm_out/merge.json --log_samples
```

```python
# Step 2 — paired audit on top of lm-eval output
from eval_trust.lm_eval_bridge import pair
from eval_trust.paired_stats import mcnemar_exact

out = pair("lm_out/merge_samples.jsonl", "lm_out/instruct_samples.jsonl")
p = mcnemar_exact(out["b"], out["c"])
print(f"n={out['n_common']}  delta={out['delta_pp']:+.1f}pp  p={p:.4f}")
```

`pair()` lines up both runs by `doc_id` and emits the (b, c) discordant
counts. Pass them to `mcnemar_exact` and you have the standard
exchange of "is this delta real?" — same workflow as Recipe 1, but
sourced from lm-eval-harness rather than custom logs.

---

## WasmAgent pipeline recipes

The following recipes use the `evomerge` package (WasmAgent trace-to-training
pipeline). Install with `pip install -e ".[dev]"`.

---

## Recipe 11: rollout traces → SFT training records

```python
from evomerge.io import load_rollouts, write_jsonl
from evomerge.pipeline.sft import to_sft_records
from evomerge.validate.schema_check import validate_training_record

branches = load_rollouts("fixtures/data-loop/rollout-branches.v1.jsonl")
sft = to_sft_records(branches, only_passing=True)  # only objective_score=1

for r in sft:
    assert validate_training_record(r).ok

write_jsonl(sft, "data/training/sft.jsonl")
print(f"{len(sft)} SFT records → data/training/sft.jsonl")
```

See also: [`examples/recipe11_rollout_to_sft.py`](examples/recipe11_rollout_to_sft.py)

---

## Recipe 12: rollout traces → DPO preference pairs

```python
from evomerge.io import load_rollouts
from evomerge.pipeline.dpo import to_dpo_records
from evomerge.validate.contamination import check_contamination

branches = load_rollouts("fixtures/data-loop/rollout-branches.v1.jsonl")
dpo = to_dpo_records(branches)

# Guard against eval-set contamination before exporting
report = check_contamination(
    training_texts=[r.chosen for r in dpo],
    eval_texts=["...your eval item texts..."],
    threshold=0.2,
)
print(f"contamination: {report.n_flagged}/{report.n_training} flagged")
assert report.n_flagged == 0
```

See also: [`examples/recipe12_rollout_to_dpo.py`](examples/recipe12_rollout_to_dpo.py)

---

## Recipe 13: ComplianceEvalRecord → answerer + repairer SFT

```python
from evomerge.schemas.compliance import (
    ComplianceEvalRecord, ConstraintViolation, ConstraintLevel,
    ConstraintCategory, RepairTraceEntry, RepairStrategy, RunMode, ViolationStage,
)
from evomerge.pipeline.compliance_sft import compliance_to_sft_records

record = ComplianceEvalRecord(
    task_id="rfi-001", task_spec_hash="abc123", model="qwen-7b",
    mode=RunMode.full_pcl, final_pass=True,
    artifact="Summary...\nAction List\n- Review by Friday",
    violations=[ConstraintViolation(
        constraint_id="c_action_list", level=ConstraintLevel.hard,
        category=ConstraintCategory.content,
        hint="Missing 'Action List' section",
        detected_at=ViolationStage.post_decode,
    )],
    repair_trace=[RepairTraceEntry(
        round=1, violation_ids=["c_action_list"],
        strategy=RepairStrategy.insert_section, ok=True,
    )],
    repair_rounds=1,
)

sft = compliance_to_sft_records([record])
print([r.output_type for r in sft])
# ['final_answer', 'repair_patch']
```

See also: [`examples/recipe13_compliance_sft.py`](examples/recipe13_compliance_sft.py)

---

## Recipe 14: A/B/C/D/E evaluation harness

```python
from evomerge.eval import EvalConfig, EvalGroup, EvalHarness, EvalRecord

TASK_IDS = [f"t{i}" for i in range(30)]
TASKS    = ["Summarise document." for _ in TASK_IDS]

def run_group_A(task_id, task):
    # replace with real model call
    i = int(task_id[1:])
    return EvalRecord(task_id, "A", final_pass=(i % 10 >= 7), latency_ms=300)

def run_group_C(task_id, task):
    i = int(task_id[1:])
    return EvalRecord(task_id, "C", final_pass=(i % 10 >= 2), latency_ms=420)

report = EvalHarness(
    EvalConfig(TASK_IDS, TASKS),
    {"A": EvalGroup("A", run_group_A), "C": EvalGroup("C", run_group_C)},
).run()

for g, m in report.metrics.items():
    print(f"{g}: pass={m.taskspec_pass_rate:.0%}  cost={m.cost_per_accepted_task:.0f} tok")
```

See also: [`examples/recipe14_eval_harness.py`](examples/recipe14_eval_harness.py)

---

## Recipe 15: prove C > A with McNemar + bootstrap

```python
from evomerge.eval import EvalRecord, paired_significance

group_A = [EvalRecord(f"t{i}", "A", i % 10 >= 7) for i in range(30)]
group_C = [EvalRecord(f"t{i}", "C", i % 10 >= 2) for i in range(30)]

r = paired_significance(group_A, group_C, label_a="A (base)", label_b="C (fine-tuned)")
print(f"delta={r.pass_rate_delta:+.1%}  p={r.mcnemar_p:.4f}  sig@05={r.significant_at_05}")
# delta=+50.0%  p=0.0000  sig@05=True
```

See also: [`examples/recipe15_significance.py`](examples/recipe15_significance.py)

---

## Recipe 16: full-loop demo (export → eval → significance → router)

```python
from evomerge.export import run_export
from evomerge.eval import EvalConfig, EvalGroup, EvalHarness, EvalRecord, paired_significance
from evomerge.router.classifier import RouterRuleClassifier
from evomerge.router.features import feature_from_record
from evomerge.synthesize.templates import TaskType, make_task_spec
import tempfile

# 1. export
with tempfile.TemporaryDirectory() as out:
    manifest = run_export(
        rollout_jsonl="fixtures/data-loop/rollout-branches.v1.jsonl",
        out_dir=out,
    )
    print(f"n_sft={manifest.n_sft}  n_dpo={manifest.n_dpo}  n_ppo={manifest.n_ppo}")

# 2. eval + significance (deterministic stubs)
ids = [f"t{i}" for i in range(40)]
A = [EvalRecord(t, "A", int(t[1:]) % 10 >= 7) for t in ids]
C = [EvalRecord(t, "C", int(t[1:]) % 10 >= 2) for t in ids]
sig = paired_significance(A, C)
print(f"p={sig.mcnemar_p:.4f}  significant={sig.significant_at_05}")

# 3. router
spec = make_task_spec(TaskType.markdown_report, intent="Summarise")
clf  = RouterRuleClassifier()
labels = [clf.predict(feature_from_record(spec, r)).value for r in C]
from collections import Counter
print(Counter(labels))
```

See also: [`examples/recipe16_full_loop_demo.py`](examples/recipe16_full_loop_demo.py)

---

## Recipe 17 — AEP evidence pipeline end-to-end demo

**File:** [`examples/recipe17_aep_trust_demo.py`](examples/recipe17_aep_trust_demo.py)

Demonstrates the complete WasmAgent evidence pipeline in 8 steps with no model, no API key, no external downloads:

1. Build synthetic AEP record (`aep/v0.2`, Ed25519-signed)
2. Validate with `validate_aep_record(require_signature=True)` — checks schema + Ed25519 signature + evidence completeness
3. Compute `AgentTrustScore` (geometric mean over up to 12 dimensions: 7 base — task_success, evidence_completeness, policy_compliance, budget_compliance, verifier_agreement, benchmark_trust, supply_chain_integrity — plus 5 optional: replay_determinism, contamination_resistance, tool_misuse_resistance, redaction_quality, runtime_isolation_level)
4. Classify benchmark exploit surfaces (`eval_trust.exploit_surface`)
5. Generate Markdown audit report (`generate_audit_report()`)
6. Build `RunReceipt` (SCITT-style provenance)
7. Register artifacts in `Registry` (aep_schema + verifier + receipt)
8. Print summary: trust_score=0.977 (grade A), 3 registry entries

```bash
python examples/recipe17_aep_trust_demo.py
```

Expected output:

```
Step 8: Summary
============================================================
  AEP validation   : PASS
  Trust score      : 0.977 (grade A)
  Exploit surfaces : 1 detected / 6 total
  Audit report     : 24 lines generated
  Run receipt      : <digest>...
  Registry entries : 3 registered

All steps completed successfully.
```

