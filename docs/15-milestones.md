# trace-pipeline — Milestones

> `trace-pipeline` (pip: `evomerge`, CLI: `evomerge …`) is the measurement-trust and
> trace-to-training backend of the WasmAgent Trustworthy Agent Training Loop. It is a
> **consumer** of the canonical `wasmagent-protocol` schemas — it must never fork them.
> Bot mode: `full`. CI is `make ci` (pytest + ruff + schema-check + reproducer + self-test + examples).

## Milestone 1 — Consume canonical schemas from `wasmagent-protocol` (retire the forks)

The AEP + compliance schemas under `schemas/` are hand-maintained forks that have drifted
from the canonical source (`wasmagent-protocol`, published as PyPI `wasmagent-protocol`).
This milestone removes the forks and depends on the package. Tracks issue #17.

- [ ] Add `wasmagent-protocol` to `pyproject.toml` dependencies (pin a released version) and expose the canonical schemas to `evomerge/validate/schema_check.py` via the installed package instead of local `schemas/*.schema.json`.
- [ ] Rewrite `scripts/sync-wasmagent-schemas.py` — its header still says "The schema SSOT lives in wasmagent-js" and maps paths under `packages/core/src/ranking/schemas/`. Point it at the `wasmagent-protocol` package (or drop it entirely if the package is a direct dependency). SSOT is `wasmagent-protocol`, not wasmagent-js.
- [ ] Delete the forked copies once consumed: `schemas/aep-record.schema.json`, `schemas/compliance-eval-record.schema.json`, `schemas/constraint-ir.schema.json`, `schemas/constraint-violation.schema.json`, `schemas/rollout-wire.schema.json`, `schemas/task-spec.schema.json`, `schemas/repair-trace-entry.schema.json` (canonical name is `repair-trace`).
  > Resolved per #17/#37/#41: only `aep-record.schema.json` was a runtime-consumed
  > fork — `evomerge/validate/aep.py` now consumes it via `get_schema("aep-record")`
  > and the retained local copy is aligned to canonical (the invented `aep/v0.4`
  > enum value is removed; there is no canonical v0.4). The other six are **not**
  > forks to delete: they are generated from this repo's own Pydantic models by
  > `scripts/export-schemas.py` (merged #22) and are the repo-local SSOT for those
  > compliance/rollout wire contracts.
- [ ] Keep repo-local schemas that are genuinely single-consumer: `schemas/dpo-training-record.schema.json`, `schemas/ppo-training-record.schema.json`, `schemas/sft-training-record.schema.json`, `schemas/trust-score.schema.json`. Document in `schemas/README` that these are trace-pipeline-owned, the rest come from the package.

## Milestone 2 — Align to canonical `aep/v0.3` (drop the rejected fork fields)

The local `aep-record` fork invented `aep/v0.3` and `aep/v0.4` fields ahead of canonical.
Per the org contract-change verdict (wasmagent-protocol#115), some are adopted upstream and
some are rejected. Align once canonical `aep/v0.3` ships.

- [ ] Consume canonical `aep/v0.3` (`recording_mode`, `side_effect_class`, `run_side_effect_class_max`, `user_id`, `subject_id`) from the package; align `evomerge/validate/aep.py` to the canonical field names/enums.
- [ ] Remove `argument_drift` (bare untyped object — rejected upstream) and `dsse_envelope` (signing belongs in the envelope layer, not inline on each record). If argument-drift metadata is still needed, propose a fully-specified sub-schema upstream in `wasmagent-protocol` first.
- [ ] Remove the local `aep/v0.4` enum value; there is no canonical v0.4.
- [ ] Update `fixtures/golden/*.jsonl` and `tests/test_schemas*.py` to validate against the canonical package schema.

## Milestone 3 — Drift gate (prevent re-forking)

- [ ] Adopt the org schema-drift CI gate (wasmagent-protocol#116) once published, or add a `make schema-check` step that fails if any local file re-declares a canonical `$id` while not depending on the package.
- [ ] Wire the gate into `make ci` so a PR that re-forks a canonical schema fails automatically.

## Milestone 4 — Data-loop contract hardening

- [ ] Keep `docs/data-loop-contract.md` (the binding three-repo contract) in sync with the canonical `rollout-wire`, `compliance-eval-record`, and training-record schemas.
- [ ] Add a round-trip test: a `ComplianceEvalRecord` produced against the canonical schema exports cleanly to SFT/DPO/PPO training records and re-validates.

```markdown
## Milestone 5 — Production Trace Pipeline at Scale

- [ ] Horizontal scaling for trace ingestion — partition AEP records by `user_id`/`subject_id` shards, implement concurrent validation workers, and add backpressure-aware queueing (Redis/RabbitMQ) to handle burst traffic from multiple agent instances.
- [ ] Streaming compliance evaluation — replace batch-mode constraint checking with incremental evaluation as AEP records arrive, enabling early detection of violations and sub-second feedback to agent control loops.
- [ ] Trace archival and retrieval layer — implement time-partitioned storage (Parquet on S3/GCS) with indexed query APIs for audit trails, reproduction, and historical trust-score calculation; add retention policies and tiered cold-storage.
- [ ] DPO/PPO/SFT training pipeline integration — wire the validated trace exporters to actual training job orchestration (Ray/Lightning), with dataset versioning, checkpoint management, and training-loss telemetry back into trust scores.
- [ ] Reproducible trace replay framework — add deterministic re-execution of recorded AEP traces with side-effect mocking and state snapshot/restore to enable "debug mode" for failed trust checks and regression testing of agent updates.
- [ ] Trust-score aggregation and serving — materialize per-subject trust scores from historical traces, serve low-latency lookups via caching layer (Redis/Memcached), and expose Prometheus metrics for score distribution and drift monitoring.
- [ ] Multi-tenant isolation — introduce tenant-scoped trace segregation (by organization/team boundaries), per-tenant resource quotas, and audit-grade access logging for compliance consumers.
- [ ] Pipeline health and SLA monitoring — add end-to-end latency tracking, per-stage success rates, alerting on validation failures/backlog buildup, and automated canary testing of schema evolution from `wasmagent-protocol` updates.
```

## Milestone 6 — Real-Time Trace Ingestion & AgentBOM-Gated Dataset Pipeline

Building on canonical schema alignment and training record generation, Milestone 6 expands `trace-pipeline` (`evomerge`) into a scalable, enterprise-grade ingestion and synthesis engine featuring stream processing, AgentBOM-informed trust scoring, secret scrubbing, and verifiable dataset lineage.

- [ ] `evomerge/ingest/stream.py` — Implement streaming AEP `v0.3` trace ingestion from blob storage and message buses with low-latency validation and dead-letter queue (DLQ) routing for corrupted records.
- [ ] `evomerge/sanitize/scrubber.py` — Add automatic PII, API key, and credential sanitization to strip sensitive payloads from tool calls and model responses prior to training dataset compilation.
- [ ] `evomerge/trust/agentbom.py` — Integrate AgentBOM compliance posture evaluations into `trust_score` calculations, penalizing trajectories that exceed declared tool permission scopes or security baselines.
- [ ] `evomerge/dataset/counterfactual.py` — Implement counterfactual repair trajectory synthesis, automatically transforming policy violations and execution failures into negative target pairs for DPO dataset generation.
- [ ] `evomerge/graph/multi_agent.py` — Build multi-agent trace graph merging to stitch parent-child subagent execution trees into unified PPO advantage training records with per-agent credit attribution.
- [ ] `evomerge/curriculum/sampler.py` — Implement dynamic curriculum sampling and hard-negative trace mining to prioritize high-leverage and policy-critical trajectories during SFT/DPO dataset assembly.
- [ ] `evomerge/lineage/ledger.py` — Emit W3C PROV-compliant dataset lineage manifests and cryptographic hash chains linking source AEP trace batches directly to compiled training output artifacts.
- [ ] `evomerge/validate/off_policy.py` — Add off-policy evaluation against `constraint-ir` schemas to verify synthetic trajectories before releasing dataset checkpoints to training runners.
- [ ] `evomerge/cli/pipeline.py` — Expose unified async pipeline commands (`evomerge pipeline run`) supporting parallelized trace ingestion, filtering, scrubbing, and export.
- [ ] `tests/test_benchmark_throughput.py` — Add performance and stress testing suite verifying target throughput (>10,000 traces/sec) under streaming ingestion workloads.
