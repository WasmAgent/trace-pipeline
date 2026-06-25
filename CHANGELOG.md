# Changelog

All notable changes to this repository.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — 2026-06-25

### Added (P2 — Trust Score + Evidence Registry)

- **`evomerge/trust_score.py`** — Agent Trust Score (P2-1). `AgentTrustScore` 综合评分（geometric
  mean of: task_success / evidence_completeness / policy_compliance / budget_compliance /
  verifier_agreement / benchmark_trust / supply_chain_integrity）。`AgentTrustScoreBuilder`
  fluent API + `compute_trust_score()` 便捷函数。Grade A–F 映射。10 tests pass。
  CLI: `python -m evomerge trust-score [--aep F] [--task-passed] [--benchmark-trust F] [--receipt F]`.

- **`evomerge/registry.py`** — Agent Evidence Registry (P2-2). 轻量 signed-JSON artifact
  registry，支持 8 种条目类型：`policy_bundle` / `verifier` / `benchmark_task` / `receipt` /
  `model_profile` / `router_profile` / `dataset_card` / `aep_schema`。SHA-256 digest tamper-evidence。
  `Registry.verify_entries()` 校验全库。7 tests pass。
  CLI: `registry-register` / `registry-list`。

### Added (P1 — cross-framework + router bridge)

- **`evomerge/benchmarks/langsmith_trace.py`** — LangSmith/LangGraph trace → AEP import.
  `LSRun` / `LSTrace` dataclasses. `ls_trace_to_aep()`: `tool` runs → `ActionEvidence`,
  chain/tool errors → `capability_decisions[deny]`, `llm` runs → `model_id`.
  Retrievers/embeddings silently skipped. `load_ls_trace_jsonl()` groups by root run.
  6 tests pass. CLI: `python -m evomerge import-langsmith --input FILE --output FILE`.

- **`evomerge/router/aep_bridge.py`** — AEP → router feature bridge (P1-3).
  `features_from_aep()` maps AEP actions/decisions/verifiers to `RouterFeatures`.
  `label_from_aep()` heuristic: 0 fails → `small_model_can_handle`, fails only → `need_repair`,
  1 deny → `need_large_model`, ≥2 denies → `need_human_review`. 6 tests pass.

- **`schemas/aep-record.schema.json` — `budget_ledger`**. Per-run budget consumption fields
  mirroring the TypeScript `BudgetLedger` added to `@wasmagent/aep`.

### Added (Day 61-90)

- **`evomerge/benchmarks/openai_agents_trace.py`** — Cross-framework trace import: OpenAI Agents SDK
  spans → AEP record (`aep/v0.1`). `OAISpan` / `OAITrace` dataclasses; `oai_trace_to_aep()` maps
  `tool.call` → `ActionEvidence`, `guardrail` → `VerifierResult`, tool errors → `capability_decisions`.
  `load_oai_trace_jsonl()` groups spans by trace_id. 6 tests pass.
  CLI: `python -m evomerge import-oai-agents --input FILE --output FILE`.

- **`evomerge/audit_report.py`** — Benchmark audit report generator. `generate_audit_report()` combines
  AEP validation, benchmark trust scores, and run provenance receipts into a Markdown report.
  `AuditReportConfig` accepts `aep_files`, `task_dirs`, `receipt_paths`. 4 tests pass.
  CLI: `python -m evomerge audit-report --title STR [--aep F]+ [--task-dir D]+ [--receipt F]+ [--output PATH]`.

### Added (Day 31-60)

- **`evomerge/benchmarks/bfcl.py`** — BFCL v4 adapter. `BFCLTask` / `BFCLFunction` / `BFCLResult` /
  `BFCLFunctionCall` dataclasses. `bfcl_to_rollout()` converts to `rollout-wire/v1`. `BFCLAdapter`
  with `load_jsonl()` / `to_rollouts()`. CLI: `python -m evomerge import-bfcl --input FILE --output FILE`.

- **`evomerge/benchmarks/mcp_atlas.py`** — MCP-Atlas adapter. `MCPAtlasTask` / `MCPAtlasClaim` /
  `MCPAtlasResult` / `MCPAtlasToolCall` dataclasses. `mcp_atlas_to_aep()` → `aep/v0.1` record.
  `mcp_atlas_to_rollout()` → `rollout-wire/v1`. `MCPAtlasAdapter` with `load_jsonl()` / `to_aep()` /
  `to_rollouts()`. CLI: `python -m evomerge import-mcp-atlas --input FILE --output FILE --format [rollout|aep]`.

- **`evomerge/provenance.py`** — Run provenance / SCITT-style receipt. `RunReceiptBuilder` fluent API
  (`add_input`, `add_output`, `add_model`, `set_policy_bundle`, `build`). `RunReceipt` with `save()`,
  `load()`, `receipt_digest` (SHA-256 of canonical JSON). Captures git HEAD, evomerge version, file digests.
  5 tests pass. CLI: `python -m evomerge receipt --run-id ID [--input F]+ [--output F]+ [--save PATH]`.

- **`evomerge/dataset_card.py` — `TraceCard`**. `TraceCard` dataclass with AEP schema version,
  run receipt path, known biases, contamination risks, excluded fields. `generate_trace_card()` renders
  structured TRACE_CARD.md. 2 tests pass (`tests/test_trace_card.py`).

### Added (P0 AEP reform)

- **`schemas/aep-record.schema.json`** — JSON Schema draft-07 for `AEPRecord` (`aep/v0.1`).
  Cross-repo source of truth for AEP validation. Required fields: `schema_version`, `run_id`,
  `created_at_ms`. Optional arrays: `actions`, `verifier_results`, `capability_decisions`,
  `input_refs`, `output_refs`.

- **`evomerge/validate/aep.py`** — AEP record validation. `validate_aep_record()` validates
  a single dict against the JSON schema (graceful fallback if `jsonschema` not installed).
  `AEPValidationResult` with `evidence_completeness` ratio and `passed` predicate.
  `validate_aep_file()` for JSONL batch validation. `print_aep_report()` for human-readable output.

- **`evomerge/__main__.py` — `validate-aep` subcommand**. Accepts `--input FILE` (JSONL) and
  `--fail-under FLOAT` (default 1.0). Exits 0 if pass rate ≥ threshold, else 1. Intended to
  gate training export: run before `python -m evomerge export`.

- **`evomerge/security/benchmark_linter.py`** — Benchmark linter v0.1. `lint_benchmark_dir()`
  runs 6 anti-reward-hacking checks on a task directory: gold-answer readability, test-script
  writability, env-file presence, grader replaceability, git-dir leak, task manifest absence.
  `BenchmarkTrustScore` with numeric score (1.0 = clean, deductions per severity) and `trusted`
  predicate. 7 tests pass (`tests/test_benchmark_linter.py`).

- **`evomerge/__main__.py` — `lint-benchmark` subcommand**. Accepts `--task-dir PATH` and
  `--fail-under FLOAT` (default 0.6). Exits 0 if trust score ≥ threshold, else 1.

### Added (P0/P1/P2 pipeline modules)

- **`evomerge/adp/export.py`** — ADP (Agent Data Protocol) export. `rollout_to_adp()` converts
  `RolloutBranchRecord` to interleaved agent/environment step list. CLI: `python -m evomerge adp-export`.

- **`evomerge/rl/export.py`** — RL transition export. `RlTransition` with `RewardSignal`
  (build / visual / policy / cost dims). CLI: `python -m evomerge rl-export --reward build,policy,cost`.

- **`evomerge/capability/taxonomy.py`** — 11-tag deterministic capability tagger (`CapabilityTag` enum,
  `tag_rollout_step()`, `tag_adp_episode()`). No ML. Rule tables are monkey-patchable for tests.

- **`evomerge/capability/attribution.py`** — Capability attribution engine. `attribute_rollout()`
  compares passing vs failing branches, identifies missing capability tags, classifies failure mode.
  `mine_capability_gaps()` aggregates across rollout lists. Outputs `CapabilityGap` with DPO/SFT suggestion.

- **`evomerge/context_compile/compiler.py`** — Trace-to-context compiler. Two modes:
  `long_context_qa` (full trace → QA record, passing branches only) and `router_critic`
  (per-step continue/stop/ask decisions). CLI: `python -m evomerge compile-context`.

- **`evomerge/security/mcp.py`** — `McpSecurityEvalRecord` Pydantic schema (`mcp-security-eval/v1`).
  Records firewall decision, risk findings, rug-pull detection, consent ref, taint signals.

- **`scripts/generate-dataset-card.py`** — Auto-generate `DATASET_CARD.md` from `manifest.json`.
  Fills template with record counts, contamination report, seed table, schema refs. `--check` mode for CI.

- **`docs/dataset-card-template.md`** — Standard dataset card template for all JSONL/model outputs.

- **CI**: Added cross-repo schema parity step (sparse-checkout wasmagent-js, runs
  `sync-wasmagent-schemas --check` + `check-schema-parity --wasmagent-js`). Dataset card smoke test.

- **Test count**: 379 tests (up from 288 at last changelog entry).

### Added (previous entries follow)

- `fixtures/golden/` — 6 golden trace fixtures covering all canonical scenarios:
  `rollout-success`, `rollout-policy-blocked`, `rollout-repair-success`,
  `compliance-direct-pass`, `compliance-direct-fail`, `compliance-repair-pass`.
- `fixtures/fixtures.lock.json` — SHA-256 hash lock for all 7 fixtures (6 golden + 1 shared data-loop).
- `scripts/verify-fixtures-lock.py` — standalone fixture hash verifier usable by any repo's CI.
- `docs/ecosystem-map.md` — canonical three-repo system diagram and terminology guide.
- CI fixture hash check step using `verify-fixtures-lock.py`.
- SFT-v2/v3 training pipeline:
  - `scripts/generate_synthetic_data_fast.py` — parallel Anthropic API synthetic data generation
    with retry on rate limits (local proxy support).
  - `scripts/merge_training_data_v2.py` — merge all training sources with full-content dedup;
    auto-imports new IFEval seeds from wasmagent-js.
  - `scripts/train_dpo.py` — DPO/ORPO fine-tuning with TRL DPOTrainer; `--merge-all` flag;
    `--force-cpu` for Apple Silicon; auto-detects LoRA vs full model checkpoint.
  - `scripts/train_sft.py` updated: auto-detects latest checkpoint for resume;
    `--force-cpu` flag (Apple Silicon only; Linux CPU does not need it).
  - `scripts/compare_base_vs_sft.py` — 3-seed A vs C comparison with McNemar + bootstrap.
  - `scripts/eval_group_ac.py` — live eval via compliance engine GGUF inference.
  - `checkpoints/base/Qwen2.5-1.5B-Instruct/` — canonical base model (hf-mirror download).
- v2 training data pipeline:
  - 1220 SFT records + 142 DPO pairs from 9 IFEval seeds + 2 synthetic batches.
  - `data/training/ifeval/` — 900 real ComplianceEvalRecords → 556 SFT + 67 repair-DPO + 34 cross-mode DPO.
  - `data/training/v2/sft_merged.jsonl` and `dpo_merged.jsonl` (gitignored, local only).
- IFEval benchmark seeds 45–50 in `wasmagent-js/packages/compliance/benchmarks/ifeval/`.
- SFT-v1 experiment results and analysis (Section 4.4 of compliance model report):
  - Null result: −4.7 pp (p=0.28) vs base model. Confirmed data scale insufficient without DPO phase.
  - GGUF Q4_K_M conversion pipeline; `evomerge-sft-compliance-v1` registered in LocalModel registry.
- README repositioned as "measurement trust + trace-to-training backend".

### Changed

- `scripts/check-schema-fields.py`: fixture hash check now uses `verify-fixtures-lock.py`.
- `evomerge/schemas/compliance.py`: `ComplianceEvalRecord` now has `schema_version` field
  (`"compliance-eval-record/v1"`) for cross-repo contract enforcement.
- `wasmagent-js/packages/model-local/src/registry.ts`: added `evomerge-sft-compliance-v1` entry.
- `wasmagent-js/packages/compliance/benchmarks/ifeval/run.ts`: supports direct GGUF path in `--model`.

## [0.2.0] — 2026-06-24

### Added — `evomerge` pipeline package

- `evomerge/schemas/` — Pydantic v2 models mirroring wasmagent-js TypeScript interfaces:
  `RolloutBranchRecord` (rollout-wire/v1), `ComplianceEvalRecord`, `TaskSpec`,
  `ConstraintViolation`, `RepairTraceEntry`, `SftTrainingRecord`, `DpoTrainingRecord`,
  `PpoTrainingRecord`, `Provenance`.
- `evomerge/pipeline/` — trace-to-training converters:
  `to_sft_records`, `to_dpo_records`, `to_ppo_records`, `compliance_to_sft_records`.
- `evomerge/io.py` — `load_jsonl`, `write_jsonl`, `write_dicts_jsonl`,
  `load_rollouts`, `load_compliance_records`, `load_router_records`.
- `evomerge/export.py` — `run_export()`: full pipeline (load → convert → validate
  → decontaminate → export sft/dpo/ppo/router/compliance_sft JSONL).
- `evomerge/validate/` — `check_contamination` (8-gram Jaccard) + `validate_training_record`.
- `evomerge/synthesize/` — built-in `TaskSpec` templates for three MVP task types
  (markdown_report, tool_call, repair) + `SyntheticGenerator` (teacher model, any API).
- `evomerge/eval/` — `EvalHarness` (A/B/C/D/E groups), `EvalMetrics` (9 metrics,
  Wilson CI), `stat_bridge` (`paired_significance`, `compare_all_groups`).
- `evomerge/router/` — `RouterFeatures` (15-dim), `RouterLabel` (4-class),
  `RouterRuleClassifier` (explainable thresholds), `build_router_records`.
- `evomerge/__main__.py` — CLI: `export`, `router`, `validate`, `synthesize`.
- `fixtures/data-loop/rollout-branches.v1.jsonl` — shared 2-branch fixture
  (byte-identical with wasmagent-js and bscode copies).
- `scripts/check-schema-fields.py` — standalone schema drift checker.
- `pyproject.toml` renamed package to `evomerge`, added `pydantic>=2.0` dependency.
- 5 new runnable examples: recipe11 (SFT), recipe12 (DPO), recipe13 (compliance SFT),
  recipe14 (eval harness), recipe15 (significance testing).
- CI expanded: schema drift check, CLI smoke (export + validate), lint covers
  `evomerge/` and `scripts/`.
- 247 new tests (288 total, up from 41).

## [Unreleased]

### Added

- Pre-arXiv preprint of the methodology paper
  *Silent Contamination in LLM Merging Evaluation* in
  `papers/eval_trust/draft.pdf`.
- `eval_trust/` audit toolkit:
  - `paired_stats.py`: McNemar exact, Wilson CI, paired bootstrap, sample-size planning.
  - `conformal_ci.py`: split-conformal accuracy intervals.
  - `t0v2/truncation_extract.py`: A_truncated channel detector.
  - `t0v2/aggregate.py`: alpha/beta/gamma routing decision aggregator.
  - `lm_eval_bridge.py`: adapter from lm-evaluation-harness samples_*.jsonl.
- `run_audit.py`: end-to-end reproducer of the case-study flip.
- `papers/eval_trust/scripts/make_figures.py`: regenerates all 3 paper figures
  from `data/`.
- `data/`: raw paired logs for the case study, anonymized marginal-protect
  history, quantization granularity summary, GSM8K dev split.
- `tests/`: 41 unit tests across paired_stats, conformal_ci, lm_eval_bridge.
- GitHub Actions CI on Python 3.10 / 3.11 / 3.12 (pytest + reproducer +
  figure regen + ruff).
- `EXAMPLES.md`: 10 copy-pasteable recipes for common audit tasks.
- `CONTRIBUTING.md`, `SECURITY.md`, issue templates.


## [0.1.0] — 2026-06-05

Initial public release. arXiv ID pending endorsement.

[Unreleased]: https://github.com/WasmAgent/trace-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/WasmAgent/trace-pipeline/releases/tag/v0.1.0
