# data/samples — Public Sample Artifacts

These files are generated from `fixtures/data-loop/rollout-branches.v1.jsonl`
and demonstrate the complete trace-to-training pipeline output.

## Files

| File | Description |
|---|---|
| `aep-sample.jsonl` | 2 AEP (`aep/v0.1`) evidence records — tool calls, verifier results, budget ledger |
| `sft.jsonl` | Supervised fine-tuning records (1 record) |
| `dpo.jsonl` | DPO chosen/rejected pairs (1 pair) |
| `ppo.jsonl` | PPO reward-labelled records (2 records) |
| `DATASET_CARD.md` | Auto-generated dataset card |
| `AUDIT_REPORT.md` | Combined provenance + lint audit report |
| `run-receipt.json` | Run provenance receipt (input/output digests) |
| `manifest.json` | Export manifest with record counts and file hashes |
| `contamination_report.json` | Benchmark contamination check results |
| `schema_report.json` | Schema validation report |
| `redaction_report.json` | PII/sensitive field redaction report |

## Regenerate

```bash
# AEP sample (hand-crafted, schema-valid aep/v0.1 records)
python3 -m evomerge validate-aep --input data/samples/aep-sample.jsonl

# Training artifacts from rollout fixture
python3 -m evomerge export \
  --rollout fixtures/data-loop/rollout-branches.v1.jsonl \
  --out-dir data/samples/

python3 -m evomerge audit-report \
  --title "WasmAgent Sample Audit" \
  --output data/samples/AUDIT_REPORT.md

python3 -m evomerge receipt \
  --run-id "sample-export-001" \
  --operator "wasmagent-sample" \
  --input fixtures/data-loop/rollout-branches.v1.jsonl \
  --output data/samples/sft.jsonl \
  --output data/samples/dpo.jsonl \
  --save data/samples/run-receipt.json
```

## Source fixture

`fixtures/data-loop/rollout-branches.v1.jsonl` — two branches of the same
task (`rollout_id: 00000000-0000-0000-0000-000000000001`):
- Branch 0: `objective_score=1` (pass) → SFT record + DPO chosen side
- Branch 1: `objective_score=0` (fail) → DPO rejected side + PPO negative
