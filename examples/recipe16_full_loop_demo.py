"""recipe16_full_loop_demo.py — end-to-end data-loop demo in one file.

Demonstrates the complete WasmAgent compliance-conditioned small model loop:

  fixture (rollout-wire/v1 JSONL)
    → run_export()  (SFT + DPO + PPO)
    → EvalHarness   (A/B/C comparison with deterministic stubs)
    → paired_significance  (McNemar + bootstrap: C > A)
    → RouterRuleClassifier  (routing predictions for group-C records)

No model, no GPU, no API key required.  Swap the run_fn stubs with real
model calls for production use.

Run:
    python examples/recipe16_full_loop_demo.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# ── 1. Export training data from the shared fixture ──────────────────────────
print("=" * 60)
print("Step 1: export rollout traces → training JSONL")
print("=" * 60)

from evomerge.export import run_export

FIXTURE = "fixtures/data-loop/rollout-branches.v1.jsonl"

with tempfile.TemporaryDirectory() as tmpdir:
    manifest = run_export(rollout_jsonl=FIXTURE, out_dir=tmpdir)
    print(json.dumps(manifest.to_dict(), indent=2))
    assert manifest.n_sft == 1
    assert manifest.n_dpo == 1
    assert manifest.n_ppo == 2
    assert manifest.n_invalid == 0
    sft_path = manifest.files["sft"]
    dpo_path = manifest.files["dpo"]

    # peek at exported records
    from evomerge.schemas.training import SftTrainingRecord, DpoTrainingRecord
    with open(sft_path) as fh:
        sft_rec = SftTrainingRecord.model_validate_json(fh.readline())
    with open(dpo_path) as fh:
        dpo_rec = DpoTrainingRecord.model_validate_json(fh.readline())

print(f"\nSFT target   : '{sft_rec.messages[-1].content[:80]}'")
print(f"DPO chosen   : '{dpo_rec.chosen[:60]}'")
print(f"DPO rejected : '{dpo_rec.rejected[:60]}'")

# ── 2. A/B/C eval harness ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: A/B/C comparison harness (deterministic stubs)")
print("=" * 60)

from evomerge.eval import EvalConfig, EvalGroup, EvalHarness, EvalRecord

TASK_IDS = [f"t{i:03d}" for i in range(40)]
TASKS    = [f"Summarise document #{i} in two sentences." for i in range(40)]

def _run_A(tid, task):
    i = int(tid[1:])
    return EvalRecord(tid, "A", final_pass=(i % 10 >= 7),
                      repair_rounds=0, latency_ms=300, prompt_tokens=80, generation_tokens=120)

def _run_B(tid, task):
    i = int(tid[1:])
    rr = 1 if i % 5 == 0 else 0
    return EvalRecord(tid, "B", final_pass=(i % 10 >= 5),
                      repair_rounds=rr, repair_rounds_ok=rr,
                      latency_ms=460, prompt_tokens=100, generation_tokens=130)

def _run_C(tid, task):
    i = int(tid[1:])
    rr = 1 if i % 8 == 0 else 0
    return EvalRecord(tid, "C", final_pass=(i % 10 >= 2),
                      repair_rounds=rr, repair_rounds_ok=rr,
                      latency_ms=420, prompt_tokens=100, generation_tokens=125)

cfg    = EvalConfig(task_ids=TASK_IDS, tasks=TASKS)
groups = {
    "A": EvalGroup("A", _run_A, "base small, direct"),
    "B": EvalGroup("B", _run_B, "base small + compliance"),
    "C": EvalGroup("C", _run_C, "fine-tuned + compliance"),
}
report = EvalHarness(cfg, groups).run()

print(f"\n{'Group':<8} {'pass%':>7} {'repair_ok%':>11} {'cost':>8} {'latency':>10}")
print("-" * 48)
for g in "ABC":
    m = report.metrics[g]
    print(f"{g:<8} {m.taskspec_pass_rate*100:>6.1f}%"
          f" {m.repair_success_rate*100:>10.1f}%"
          f" {m.cost_per_accepted_task:>8.0f}"
          f" {m.latency_per_accepted_ms:>9.0f}ms")

# ── 3. Statistical significance ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: McNemar + bootstrap — is C > A significant?")
print("=" * 60)

from evomerge.eval import paired_significance, compare_all_groups

group_records = {
    g: [_run_A(tid, t) if g == "A" else
        _run_B(tid, t) if g == "B" else
        _run_C(tid, t)
        for tid, t in zip(TASK_IDS, TASKS)]
    for g in "ABC"
}

sig = paired_significance(group_records["A"], group_records["C"],
                          label_a="A (base)", label_b="C (fine-tuned)")
print(f"\n  delta          : +{sig.pass_rate_delta:.1%}")
print(f"  McNemar p      : {sig.mcnemar_p:.4f}")
print(f"  significant@05 : {sig.significant_at_05}")
print(f"  bootstrap CI   : [{sig.bootstrap['ci_lo']:.2f}, {sig.bootstrap['ci_hi']:.2f}]")

all_vs_a = compare_all_groups(group_records, reference="A")
print("\n  All groups vs A:")
for key, r in all_vs_a.items():
    marker = "✓" if r.significant_at_05 else " "
    print(f"    {key:<8}  delta={r.pass_rate_delta:+.1%}  p={r.mcnemar_p:.4f}  {marker}")

assert sig.significant_at_05, "C > A should be significant with n=40"

# ── 4. Router predictions for group-C records ────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: RouterRuleClassifier — routing group-C outcomes")
print("=" * 60)

from evomerge.router.classifier import RouterRuleClassifier
from evomerge.router.features import feature_from_record
from evomerge.router.labels import RouterLabel, label_from_record
from evomerge.synthesize.templates import TaskType, make_task_spec

spec = make_task_spec(TaskType.markdown_report, intent="Summarise document")
clf  = RouterRuleClassifier()

c_records = group_records["C"]
label_counts: dict[str, int] = {}
for rec in c_records:
    feat  = feature_from_record(spec, rec)
    label = clf.predict(feat)
    label_counts[label.value] = label_counts.get(label.value, 0) + 1

print("\n  Routing distribution for group-C (40 tasks):")
for label, count in sorted(label_counts.items()):
    bar = "█" * count
    print(f"    {label:<30} {count:>3}  {bar}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Full loop demo complete")
print("=" * 60)
print(f"  SFT records exported   : {manifest.n_sft}")
print(f"  DPO pairs exported     : {manifest.n_dpo}")
print(f"  Group-C pass rate      : {report.metrics['C'].taskspec_pass_rate:.0%}")
print(f"  A vs C McNemar p       : {sig.mcnemar_p:.4f}  (significant: {sig.significant_at_05})")
print(f"  Router label variety   : {len(label_counts)} distinct labels")
