# Trace to Training Data in 10 Minutes

This tutorial takes a `rollout-wire/v1` JSONL file — exported from bscode or
produced by wasmagent-js — and converts it into SFT/DPO training records, a
dataset card, and an audit report.

No GPU or API key required. All steps use the `evomerge` CLI.

---

## Prerequisites

```bash
pip install evomerge
evomerge --version   # should print 0.2.x
```

Or from source:

```bash
git clone https://github.com/WasmAgent/trace-pipeline
cd trace-pipeline
pip install -e ".[dev]"
```

---

## Step 0 — Get a rollout JSONL file

**Option A — from bscode (recommended)**

```bash
curl -s "https://<your-worker>.workers.dev/rollouts/export" \
  -H "X-Session-Id: <session>" \
  > rollouts.jsonl
```

**Option B — use the bundled fixture**

```bash
cp fixtures/data-loop/rollout-branches.v1.jsonl rollouts.jsonl
```

**Option C — from wasmagent-js CLI**

```bash
wasmagent export-rollouts --in ranked.jsonl --format rollout-wire --out rollouts.jsonl
```

---

## Step 1 — Validate the schema

```bash
evomerge validate rollouts.jsonl
```

Expected output:

```
✓  3 records parsed
✓  schema_version: rollout-wire/v1
✓  0 redaction violations
✓  quality gate: pass
```

If the quality gate fails, inspect the report:

```bash
evomerge validate rollouts.jsonl --verbose
```

---

## Step 2 — Export training records

```bash
evomerge export rollouts.jsonl --out ./training/
```

This creates:

```
training/
  sft.jsonl          # supervised fine-tuning records
  dpo.jsonl          # DPO chosen/rejected pairs
  router.jsonl       # routing classifier labels
  manifest.json      # counts + file hashes
```

Peek at the DPO pairs:

```bash
python3 - <<'PY'
import json
with open("training/dpo.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print("chosen  :", r["chosen"][:100])
        print("rejected:", r["rejected"][:100])
        print()
PY
```

Export a specific format only:

```bash
evomerge export rollouts.jsonl --format sft  --out training/sft.jsonl
evomerge export rollouts.jsonl --format dpo  --out training/dpo.jsonl
```

---

## Step 3 — Generate a dataset card

```bash
evomerge dataset-card rollouts.jsonl --out DATASET_CARD.md
cat DATASET_CARD.md
```

The card includes: record counts, schema version, date range, model provider
distribution, objective score distribution, and redaction summary.

---

## Step 4 — Generate an audit report

```bash
evomerge audit-report rollouts.jsonl --out AUDIT_REPORT.md
cat AUDIT_REPORT.md
```

The report includes: provenance chain, policy bundle digest, tool manifest
digest, contamination check results, and trust score.

---

## Step 5 — Validate AEP evidence records (optional)

If your rollouts include `aep_record` fields (emitted by `@wasmagent/aep`):

```bash
evomerge validate-aep rollouts.jsonl
evomerge trust-score  rollouts.jsonl
```

---

## Step 6 — Run significance tests on your eval results (optional)

If you have A/B eval results and want to know whether the improvement is real:

```python
from eval_trust.paired_stats import paired_mcnemar, wilson_ci

# a_correct[i] = True if model A got task i right
result = paired_mcnemar(a_correct, b_correct)
print(f"delta = {result.pass_rate_delta:+.1%}")
print(f"McNemar p = {result.mcnemar_p:.4f}")
print(f"significant@0.05: {result.significant_at_05}")
```

→ [eval_trust API reference](../eval_trust/)

---

## Full pipeline in one script

```bash
#!/usr/bin/env bash
set -e

INPUT=rollouts.jsonl
OUT=./output

evomerge validate "$INPUT"
evomerge export   "$INPUT" --out "$OUT/"
evomerge dataset-card "$INPUT" --out "$OUT/DATASET_CARD.md"
evomerge audit-report "$INPUT" --out "$OUT/AUDIT_REPORT.md"

echo ""
echo "Done. Output files:"
ls -lh "$OUT/"
```

---

## What the records look like

**SFT record** (`training/sft.jsonl`):

```json
{
  "messages": [
    {"role": "user",      "content": "Write a function that reverses a string."},
    {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}
  ],
  "metadata": {
    "source": "rollout-wire/v1",
    "objective_score": 1,
    "compliance_pass": true,
    "run_id": "run-abc123"
  }
}
```

**DPO record** (`training/dpo.jsonl`):

```json
{
  "prompt":   "Write a function that reverses a string.",
  "chosen":   "def reverse_string(s):\n    return s[::-1]",
  "rejected": "def reverse(s):\n    result = ''\n    for c in s: result = c + result\n    return result",
  "metadata": {
    "chosen_score":   1,
    "rejected_score": 0,
    "run_id":         "run-abc123"
  }
}
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `schema_version mismatch` | File is not `rollout-wire/v1` | Use `wasmagent validate-rollouts` to check the source |
| `quality gate: fail — unknown objective_score` | Records have no build result | Only records with `objective_score` 0 or 1 produce DPO pairs |
| `0 dpo pairs exported` | All branches have the same score | DPO requires at least one chosen/rejected pair per task |
| `contamination warning` | Overlap with known benchmarks detected | Review `AUDIT_REPORT.md` contamination section |

---

## Next steps

- [bscode DEMO_SCRIPT.md](https://github.com/WasmAgent/bscode/blob/main/docs/DEMO_SCRIPT.md) — produce real rollout data from bscode
- [wasmagent-js rollout schema](https://github.com/WasmAgent/wasmagent-js/blob/main/packages/core/src/ranking/schemas/rollout-wire.schema.json)
- [evomerge CLI reference](../README.md#cli)
- [eval_trust significance tests](../eval_trust/)
