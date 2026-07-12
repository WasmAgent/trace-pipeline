# WasmAgent Ecosystem Map

> Single source of truth for how the three repositories form a Trustworthy Agent Training Loop.
> Each repository README should reference this document and display the diagram relevant to its own layer.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│ User / TaskSpec / Enterprise Policy                           │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ wasmagent-js                                                  │
│ Runtime Compliance Source of Truth                            │
│ - Kernel / Sandbox / Remote Execution                         │
│ - CapabilityManifest / Policy / Guardrails                    │
│ - Verifier / RepairTrace / ComplianceEvalRecord emitter       │
│ - Framework adapters: MCP / AI SDK / Mastra / OpenAI Agents   │
└───────────────────────────────┬──────────────────────────────┘
                                │ verified rollout / trace / evidence
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ bscode                                                        │
│ Real Workload + Evidence Collection Surface                   │
│ - Cloudflare reference deployment                             │
│ - Visual verifier / build result / job metadata               │
│ - User-visible demo + controlled trajectory export            │
└───────────────────────────────┬──────────────────────────────┘
                                │ sanitized JSONL / objective signals
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ trace-pipeline                                            │
│ Measurement Trust + Trace-to-Training Backend                 │
│ - eval_trust: benchmark audit / paired statistics             │
│ - evomerge: schema validation / SFT / DPO / router records    │
│ - data quality gate / contamination checks / provenance       │
└───────────────────────────────┬──────────────────────────────┘
                                │ training + audit feedback
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ Better runtime policy / verifier / router / small model        │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                └────────── back to wasmagent-js
```

---

## Repository Roles

| Repository | Role | What it is NOT |
|---|---|---|
| `wasmagent-js` | Runtime compliance source of truth | Not a generic agent framework |
| `bscode` | Real workload and evidence collection surface | Not a general-purpose IDE |
| `trace-pipeline` | Measurement trust + trace-to-training backend | Not a generic ML training platform |

---

## Data Contract Flow

```
bscode
  produces → rollout-wire/v1 JSONL (via /rollouts/export)
           → ComplianceEvalRecord JSONL (via compliance engine)

wasmagent-js
  defines  → Schema SSOT (rollout-wire.schema.json, compliance-eval-record.schema.json)
  ranks    → RolloutRanker (objective_score, rank, total_score)

trace-pipeline
  consumes → ranked rollout JSONL → SFT / DPO / PPO records
  consumes → ComplianceEvalRecord JSONL → compliance SFT / DPO / router records
  validates → schema parity, contamination, provenance

Schema authority:
  wasmagent-js/packages/core/src/ranking/RolloutExporter.ts  (defines wire types)
  wasmagent-js/packages/core/src/ranking/schemas/            (JSON Schema SSOT)

  Downstream consumers in trace-pipeline:
    evomerge/pipeline/sft.py      — rollout → SFT records
    evomerge/pipeline/dpo.py      — rollout → DPO records
    evomerge/pipeline/ppo.py      — rollout → PPO records
    evomerge/schemas/rollout.py   — Pydantic mirror of rollout-wire.schema.json
    evomerge/schemas/training.py  — Pydantic mirror of training record schemas

  Note: upstream supports fork-level DPO via RolloutTreeExporter.toDpoRecordWithForkContext()
  (emits RolloutTreeRecord). Downstream currently skips these records gracefully.
```

**Schema SSOT locations:**
- `wasmagent-js/packages/core/src/ranking/schemas/rollout-wire.schema.json`
- `wasmagent-js/packages/compliance/schemas/compliance-eval-record.schema.json`
- `trace-pipeline/schemas/*.schema.json` (Pydantic mirrors, validated in CI)

---

## Shared Fixtures

All three repositories share byte-identical fixture files for smoke tests and CI.
See `trace-pipeline/fixtures/fixtures.lock.json` for SHA-256 hashes.

**Golden fixtures** (`trace-pipeline/fixtures/golden/`):

| Fixture | Scenario |
|---|---|
| `rollout-success.v1.jsonl` | Single passing rollout branch |
| `rollout-policy-blocked.v1.jsonl` | Task blocked by capability policy |
| `rollout-repair-success.v1.jsonl` | 2-branch rollout: fail → repair → pass |
| `compliance-direct-pass.v1.jsonl` | Direct mode, first attempt passes |
| `compliance-direct-fail.v1.jsonl` | Direct mode, violates constraint |
| `compliance-repair-pass.v1.jsonl` | full_pcl mode, repair resolves violation |

**Shared data-loop fixture** (`fixtures/data-loop/rollout-branches.v1.jsonl`):
Byte-identical across wasmagent-js, trace-pipeline, and bscode.

---

## Canonical Terminology

| Use this | Not this |
|---|---|
| WasmAgent Runtime | generic agent framework |
| Runtime Compliance Layer | ad-hoc repair log |
| bscode reference deployment | IDE / coding product |
| Real workload and evidence collection | demo / playground |
| evomerge measurement-trust backend | generic trainer |
| trace-to-training pipeline | random dataset export |
| `rollout-wire/v1` | trajectory format (various) |
| `ComplianceEvalRecord` | compliance log / repair log |
| golden fixture | sample data / toy data |

---

## Cross-Repo CI Contract

```bash
# Run in trace-pipeline CI:
python scripts/check-schema-fields.py        # field coverage vs data-loop contract
python scripts/export-schemas.py --check     # Pydantic → JSON Schema parity
python scripts/check-schema-parity.py \      # fixture validates against both schemas
  --runs-dir /path/to/wasmagent-js/...

# Fixture hash check:
python -c "
import hashlib, json
lock = json.load(open('fixtures/fixtures.lock.json'))
for e in lock['fixtures']:
    actual = hashlib.sha256(open(e['fixture'],'rb').read()).hexdigest()
    assert actual == e['sha256'], f'DRIFT: {e[\"fixture\"]}'
print('all fixture hashes OK')
"
```

---

*Last updated: 2026-06-25. Maintained in `trace-pipeline/docs/ecosystem-map.md`.*
