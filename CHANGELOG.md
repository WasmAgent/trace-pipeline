# Changelog

All notable changes to this repository.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- 185 new tests (226 total, up from 41).

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

[Unreleased]: https://github.com/telleroutlook/evomerge-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/telleroutlook/evomerge-framework/releases/tag/v0.1.0
