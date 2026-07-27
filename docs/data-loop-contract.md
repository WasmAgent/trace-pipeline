# Data-Loop Contract — Binding Three-Repository Agreement

> **Status: BINDING.** This document is the versioned contract for the WasmAgent
> Trustworthy Agent Training Loop. The three repositories named below MUST agree on the
> schema versions and required fields declared here. `scripts/check-schema-fields.py`
> derives its required-field sets from this contract and runs in `make schema-check` / CI,
> so a drift between this document and the canonical schemas fails the build.
>
> **Scope.** This contract covers only the **data-loop schemas**: the runtime evidence
> wire format (`rollout-wire`), the compliance verdict format (`compliance-eval-record`),
> and the trace-to-training output formats (`sft` / `dpo` / `ppo` training records). It is
> the authoritative cross-repo reference for those five schemas. AEP evidence
> (`aep-record`) is governed separately by `wasmagent-protocol`; see
> [docs/ecosystem-map.md](ecosystem-map.md) for the broader loop.

---

## 1. The three repositories

| Repository | Role in the loop | Bound by this contract |
|---|---|---|
| `wasmagent-js` | Runtime compliance source of truth. **Produces** `rollout-wire/v1` branches (via `RolloutForkRunner` / `RolloutExporter`) and `compliance-eval-record/v1` verdicts (compliance engine). Ranks branches (`objective_score`, `rank`, `total_score`). | Emits every required field of §3.1 and §3.2. |
| `bscode` | Real-workload evidence collection surface. **Produces** `rollout-wire/v1` JSONL (via `/rollouts/export`) and compliance records from the reference deployment. | Emits every required field of §3.1; carries the shared data-loop fixture byte-identical. |
| `trace-pipeline` | Measurement-trust + trace-to-training backend. **Consumes** `rollout-wire/v1` and `compliance-eval-record/v1`, **validates** them, and **exports** `sft/v1`, `dpo/v1`, `ppo/v1` training records. | Accepts nothing less than the required fields in §3; emits every required field of §3.3–§3.5. Owns the training-record SSOT. |

**Schema authority.** Per the org schema-governance verdict (Milestone 1, issue #17):
the cross-repo schemas (`rollout-wire`, `compliance-eval-record`) are canonical in
`wasmagent-protocol` (PyPI: `wasmagent-protocol`); `wasmagent-js` is the runtime emitter,
not the schema SSOT. The `*-training-record` schemas are single-consumer and remain the
SSOT in `trace-pipeline/schemas/`. The copies under `trace-pipeline/schemas/*.schema.json`
are the in-tree mirrors that this contract's field lists are verified against in CI
(`scripts/export-schemas.py --check` confirms the Pydantic models match them).

---

## 2. Canonical schema versions

These are the only `schema_version` values permitted on the loop today:

| Schema | `schema_version` | Canonical home | trace-pipeline mirror |
|---|---|---|---|
| Rollout wire | `rollout-wire/v1` | `wasmagent-protocol` | `schemas/rollout-wire.schema.json` |
| Compliance eval record | `compliance-eval-record/v1` | `wasmagent-protocol` | `schemas/compliance-eval-record.schema.json` |
| SFT training record | `sft/v1` | `trace-pipeline` (SSOT) | `schemas/sft-training-record.schema.json` |
| DPO training record | `dpo/v1` | `trace-pipeline` (SSOT) | `schemas/dpo-training-record.schema.json` |
| PPO/GRPO training record | `ppo/v1` | `trace-pipeline` (SSOT) | `schemas/ppo-training-record.schema.json` |

A new major version (e.g. `rollout-wire/v2`) is a contract change: it requires a
coordinated PR across the bound repositories and a bump in §2 + §3 of this document.
Consumers MUST reject records whose `schema_version` is not listed here.

---

## 3. Required-field contracts

The field sets below are the **minimum a producer must emit and a consumer may rely on**.
They are copied verbatim from the `required` arrays of the canonical JSON Schemas; the
standalone checks in `scripts/check-schema-fields.py::_standalone_checks` enforce exactly
these sets. If you edit a schema's `required` array, edit the matching set here in the
same PR — the CI gate reads this contract as the source of truth.

### 3.1 `rollout-wire/v1` — `RolloutBranchRecord`

One branch produced by `wasmagent-js` `RolloutForkRunner`. One JSONL line = one branch.

```
schema_version      const "rollout-wire/v1"
rollout_id          string   — groups branches belonging to the same task
task                string   — the prompt / task spec
branch_index        integer  — 0-based index within the rollout
temperature         number   — sampling temperature for this branch
session_id          string   — runtime session that produced the branch
tool_call_sequence  array    — ordered tool-call entries (may be empty)
final_answer        string   — the branch's final assistant answer
```

Constrained optional fields producers should populate when available:
`objective_status ∈ {pass, fail, unknown}`, `objective_score ∈ {0, 1}`,
`build_result.status ∈ {pass, fail, skip}`, `rank`, `total_score`, `seed`.
DPO export requires at least one `objective_score=1` and one `objective_score=0` branch
within the same `rollout_id`.

### 3.2 `compliance-eval-record/v1` — `ComplianceEvalRecord`

Final output of one WasmAgent compliance-engine run.

```
schema_version   const "compliance-eval-record/v1"
task_id          string
task_spec_hash   string   — hash of the TaskSpec this run was evaluated against
model            string   — model identifier
mode             enum {direct, prompt_retry, full_pcl}
final_pass       boolean  — whether the final artifact satisfies all hard constraints
artifact         string   — the final produced artifact
```

Optional but expected: `violations[]`, `repair_trace[]`, `repair_rounds`, `token_cost`,
`latency_ms`, `error`. Compliance-derived training records (see
`evomerge/pipeline/compliance_sft.py`, `compliance_dpo.py`) are keyed on `task_id` +
`final_pass`.

### 3.3 `sft/v1` — `SftTrainingRecord`

```
schema_version  const "sft/v1"
messages        array    — full conversation; last assistant turn is the training target
output_type     enum {final_answer, repair_patch, tool_call, next_action, escalation}
provenance      object   — { source, rollout_id?, task_id?, n_gram_hash?, task_hash? }
```

### 3.4 `dpo/v1` — `DpoTrainingRecord`

```
schema_version  const "dpo/v1"
messages        array    — full conversation with `chosen` as the final assistant turn
chosen          string   — the preferred assistant response
rejected        string   — the dispreferred assistant response
provenance      object   — { source, rollout_id?, task_id?, n_gram_hash?, task_hash? }
```

Optional: `prompt_messages` (messages without the final assistant turn, for TRL).

### 3.5 `ppo/v1` — `PpoTrainingRecord`

```
schema_version  const "ppo/v1"
messages        array    — full conversation
reward          number   — normalised to [0, 1]
provenance      object   — { source, rollout_id?, task_id?, n_gram_hash?, task_hash? }
```

`provenance.source` MUST be set on every training record so the exported dataset remains
auditable back to its producing rollout/compliance run.

---

## 4. Data-loop flow

```
wasmagent-js / bscode
   produce  →  rollout-wire/v1 JSONL  (one line per branch)
            →  compliance-eval-record/v1 JSONL  (one line per compliance run)
                    │
                    ▼
trace-pipeline
   validate →  evomerge validate  (schema + quality gate + redaction)
   export   →  sft/v1, dpo/v1, ppo/v1 JSONL  (from rollout-wire)
            →  compliance SFT / DPO / router records  (from compliance-eval-record)
   audit    →  dataset card, audit report, trust score
                    │
                    ▼
   training + audit feedback → better runtime policy / verifier / router → wasmagent-js
```

The shared smoke-test fixture `fixtures/data-loop/rollout-branches.v1.jsonl` (2 branches:
one `pass`, one `fail`) is byte-identical across the three repositories — its hash is
locked in `fixtures/fixtures.lock.json` and its expected export counts
(`n_sft=1, n_dpo=1, n_ppo=2`) are asserted by `tests/test_integration_fixture.py`.

---

## 5. Keeping this contract in sync (the drift gate)

This document is enforced, not merely consulted. The following CI steps fail when it
drifts from the canonical schemas:

```bash
make schema-check                           # runs both checks below
python scripts/check-schema-fields.py       # required-field sets (§3) vs Pydantic models
python scripts/export-schemas.py --check    # Pydantic models vs schemas/*.schema.json
```

- `check-schema-fields.py::_standalone_checks` mirrors the §3 required-field sets. If a
  schema gains or loses a required field, update both the schema and the matching §3 block
  in the same PR, or the gate reports `DRIFT`.
- When comparing against a checkout of the upstream schema homes, pass
  `--wasmagent-js PATH` (or, post-Milestone 1, the `wasmagent-protocol` package paths) to
  diff the canonical `required` arrays directly.

---

## 6. Change protocol

1. **Field addition (optional).** Adding an optional field to a schema needs no contract
   bump; update the schema, the Pydantic mirror, and `scripts/export-schemas.py --check`.
2. **Required-field change.** Editing any `required` array is a contract change: update
   the matching §3 set here, the canonical schema, the Pydantic mirror, and any fixtures
   / golden records in one coordinated PR across the bound repositories.
3. **Version bump.** A new `schema_version` (e.g. `/v2`) requires a new row in §2, a new
   §3 subsection, consumer support for both versions during the transition, and a
   fixture under `fixtures/data-loop/` pinned in `fixtures/fixtures.lock.json`.
4. **Shared fixture change.** Any change to `fixtures/data-loop/rollout-branches.v1.jsonl`
   must land in `wasmagent-js`, `bscode`, and `trace-pipeline` simultaneously and update
   the SHA-256 in `fixtures/fixtures.lock.json` (see `manifest.json::sync_note`).

---

## 7. Related documents

- [docs/ecosystem-map.md](ecosystem-map.md) — system overview, repository roles, canonical terminology.
- [docs/TRACE_TO_TRAINING_10MIN.md](TRACE_TO_TRAINING_10MIN.md) — end-to-end walkthrough from `rollout-wire/v1` to training records.
- [docs/15-milestones.md](15-milestones.md) — Milestone 4 (data-loop contract hardening) tracks this document; Milestone 1 covers the canonical-schema migration from `wasmagent-protocol`.

---

*Last updated: 2026-07-27. Maintained in `trace-pipeline/docs/data-loop-contract.md`; the
bound repositories reference this file as the data-loop SSOT.*
