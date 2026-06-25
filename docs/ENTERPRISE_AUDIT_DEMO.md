# Enterprise Audit Demo

This walkthrough shows how to build a complete, immutable audit trail for an
agent run using five evomerge CLI commands. No model or API key required —
all steps run against the bundled sample artifacts.

**Audience:** enterprise security teams, compliance officers, MLOps engineers.

**Outcome:** a chain-of-custody record from agent run → evidence validation →
trust score → signed receipt → registry entry → audit report.

---

## Prerequisites

```bash
git clone https://github.com/WasmAgent/trace-pipeline
cd trace-pipeline
pip install -e "."
```

Verify:

```bash
python3 -m evomerge --help
```

---

## The five-command chain

```text
AEP records      (from @wasmagent/aep)
    ↓
validate-aep     — schema check, required fields, AEP version
    ↓
trust-score      — composite [0.0–1.0] score with per-dimension breakdown
    ↓
receipt          — SHA-256 provenance receipt: inputs, outputs, model, policy digest
    ↓
registry-register — append to tamper-evident Evidence Registry
    ↓
audit-report     — combined Markdown report for human review
```

---

## Step 1 — validate-aep

The sample rollout fixture does not contain AEP records (it is a
`rollout-wire/v1` file). In a real deployment, AEP records are emitted by
`@wasmagent/aep` and attached to the rollout. For this demo we validate the
rollout schema directly.

```bash
# Validate the rollout-wire/v1 fixture (schema + quality gate)
python3 -m evomerge validate \
  --rollout fixtures/data-loop/rollout-branches.v1.jsonl

# In production with AEP records attached:
# python3 -m evomerge validate-aep --aep path/to/aep-records.jsonl
```

Expected output:

```
[ok] 2 rollout records parsed
[ok] schema_version: rollout-wire/v1
[ok] 0 redaction violations
[ok] quality gate: pass
```

---

## Step 2 — trust-score

```bash
python3 -m evomerge trust-score \
  --rollout fixtures/data-loop/rollout-branches.v1.jsonl
```

Expected output (approximate):

```json
{
  "overall": 0.75,
  "grade": "B",
  "breakdown": {
    "task_success": 0.5,
    "evidence_completeness": 1.0,
    "policy_compliance": 1.0
  },
  "notes": ["1/2 branches pass objective scoring"]
}
```

**Interpretation:**
- `overall ≥ 0.9` (grade A) → safe for DPO/PPO training
- `overall ≥ 0.75` (grade B) → safe for SFT training with review
- `overall < 0.6` (grade D/F) → do not use; investigate before re-export

---

## Step 3 — receipt

A `RunReceipt` binds the pipeline run to its exact inputs and outputs via
SHA-256 digests. Inspired by [SCITT](https://scitt.io/) / in-toto, without
full PKI.

```bash
# Export training data first
mkdir -p /tmp/audit-demo
python3 -m evomerge export \
  --rollout fixtures/data-loop/rollout-branches.v1.jsonl \
  --out-dir /tmp/audit-demo/

# Produce the receipt
python3 -m evomerge receipt \
  --run-id   "audit-demo-$(date +%Y%m%d-%H%M%S)" \
  --operator "security-team" \
  --input  fixtures/data-loop/rollout-branches.v1.jsonl \
  --output /tmp/audit-demo/sft.jsonl \
  --output /tmp/audit-demo/dpo.jsonl \
  --save   /tmp/audit-demo/run-receipt.json

cat /tmp/audit-demo/run-receipt.json
```

Key fields in the receipt:

| Field | Purpose |
|---|---|
| `run_id` | Unique identifier for this export run |
| `repo_commit` | Git HEAD SHA — ties the pipeline version to the code |
| `inputs[*].digest` | SHA-256 of each input file |
| `outputs[*].digest` | SHA-256 of each output file |
| `receipt_digest` | Self-referential digest of the receipt JSON |

---

## Step 4 — registry-register

The Evidence Registry is a local append-only log. Each registered artifact
gets a monotonic sequence number and a timestamp.

```bash
python3 -m evomerge registry-register \
  --artifact  /tmp/audit-demo/run-receipt.json \
  --label     "audit-demo rollout export" \
  --registry  /tmp/audit-demo/registry.json

python3 -m evomerge registry-list \
  --registry  /tmp/audit-demo/registry.json
```

Expected `registry-list` output:

```
seq  registered_at              label                          artifact
1    2026-06-25T10:00:00Z       audit-demo rollout export      run-receipt.json
```

---

## Step 5 — audit-report

The audit report combines schema validation, provenance, and any AEP or lint
findings into a single Markdown document.

```bash
python3 -m evomerge audit-report \
  --title   "Enterprise Audit Demo — $(date +%Y-%m-%d)" \
  --receipt /tmp/audit-demo/run-receipt.json \
  --output  /tmp/audit-demo/AUDIT_REPORT.md

cat /tmp/audit-demo/AUDIT_REPORT.md
```

---

## One-shot script

```bash
#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"
FIXTURE=fixtures/data-loop/rollout-branches.v1.jsonl
OUT=/tmp/audit-demo
RUN_ID="enterprise-demo-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT"

echo "=== Step 1: validate ==="
python3 -m evomerge validate --rollout "$FIXTURE"

echo "=== Step 2: trust-score ==="
python3 -m evomerge trust-score --rollout "$FIXTURE"

echo "=== Step 3: export ==="
python3 -m evomerge export --rollout "$FIXTURE" --out-dir "$OUT/"

echo "=== Step 4: receipt ==="
python3 -m evomerge receipt \
  --run-id "$RUN_ID" --operator "security-team" \
  --input  "$FIXTURE" \
  --output "$OUT/sft.jsonl" --output "$OUT/dpo.jsonl" \
  --save   "$OUT/run-receipt.json"

echo "=== Step 5: registry ==="
python3 -m evomerge registry-register \
  --artifact "$OUT/run-receipt.json" \
  --label    "enterprise-demo" \
  --registry "$OUT/registry.json"

echo "=== Step 6: audit report ==="
python3 -m evomerge audit-report \
  --title "Enterprise Audit Demo" \
  --receipt "$OUT/run-receipt.json" \
  --output "$OUT/AUDIT_REPORT.md"

echo ""
echo "Artifacts in $OUT:"
ls -lh "$OUT/"
```

---

## What to look for in the audit report

| Section | Red flag |
|---|---|
| Schema validation | Any `[FAIL]` entries |
| Contamination check | Overlap > 0% with known benchmarks |
| Redaction | Any `pii_fields_found > 0` without operator approval |
| Trust score | `overall < 0.6` (grade D or F) |
| Receipt digest | Mismatch between saved digest and recomputed digest |

---

## Pre-built sample

The `data/samples/` directory contains pre-generated artifacts from the
bundled fixture:

```
data/samples/
  sft.jsonl            # 1 SFT record
  dpo.jsonl            # 1 DPO pair
  DATASET_CARD.md      # auto-generated dataset card
  AUDIT_REPORT.md      # audit report (no AEP records — fixture only)
  run-receipt.json     # provenance receipt
  manifest.json        # export manifest
```

→ [data/samples/README.md](../data/samples/README.md)

---

## Related

- [TRACE_TO_TRAINING_10MIN.md](./TRACE_TO_TRAINING_10MIN.md) — end-to-end training data tutorial
- [bscode DEMO_SCRIPT.md](https://github.com/WasmAgent/bscode/blob/main/docs/DEMO_SCRIPT.md) — produce real rollout data from a running bscode instance
- [`@wasmagent/aep`](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/aep) — emit AEP evidence records from your agent
