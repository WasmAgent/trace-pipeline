# trace-pipeline — Development Guide for Claude

## What this project is (and is not)

**Is:** Measurement trust and trace-to-training backend for WasmAgent. Validates AEP evidence, audits benchmark claims, scores training eligibility, exports SFT/DPO/router records.

**Is NOT — do not implement these:**
- A training framework (TRL / Axolotl / Unsloth territory) — we produce training *data*, not training *infrastructure*
- A compliance certification tool — `trust_score` and `compute_calibrated_trust_score` are health scores, never claim they satisfy EU AI Act / ISO 42001
- A general observability platform (LangSmith / Phoenix territory)
- A standalone eval benchmark — eval adapters (BFCL, τ-bench, AgentHarm) import results, not reproduce them

## Test Commands

```bash
# Install (with dev deps)
pip install -e ".[dev]"

# Run all tests
PYTHONPATH=. pytest tests/ -q

# Run a specific file
PYTHONPATH=. pytest tests/test_quality_gate.py -v

# Run linter
python -m ruff check evomerge/
python -m ruff check --fix evomerge/   # auto-fix
```

## CLI

```bash
evomerge validate-aep --input data/smoke/aep-smoke.jsonl
evomerge audit-report --aep data/smoke/aep-smoke.jsonl --output AUDIT_REPORT.md
evomerge export --input data/smoke/rollout-smoke.jsonl --format sft --output sft.jsonl
wasmagent-trace validate-aep ...   # alias for evomerge
```

## Key modules (2026-06-26)

| Module | Location |
|---|---|
| `compute_admission_score` / `admission_gate` | `evomerge/validate/quality_gate.py` — 6-dimension Evidence Admission Score |
| `validate_aep_record` / `validate_aep_file` | `evomerge/validate/aep.py` — AEP v0.1/v0.2 schema validation |
| `generate_audit_report` / `_standards_section` | `evomerge/audit_report.py` — OWASP/OTel standards coverage matrix |
| `audit_verifier_results` / `audit_aep_verifiers` | `evomerge/eval/verifier_audit.py` — verifier strength audit |
| `compute_calibrated_trust_score` | `evomerge/trust_score.py` — 3-dimension calibrated score (evidence_health / policy_risk / training_eligibility) |
| `generate_aep_dataset_card` | `evomerge/dataset_card.py` — AEP dataset card with consent/redaction/admission |
| `compute_benchmark_ci` / `benchmark_ci_report` | `evomerge/conformal_ci.py` — Wilson CI for benchmark reports |
| `bfcl_to_rollout` | `evomerge/benchmarks/bfcl.py` |
| `tau_bench_to_rollout` | `evomerge/benchmarks/tau_bench.py` |
| Paper appendix generator | `papers/eval_trust/scripts/generate_appendix.py` |

## AEP schema version

Current schema: `aep-record/v0.2` — supports v0.1 and v0.2 records (backwards compatible).
Schema file: `schemas/aep-record.schema.json`

v0.2 adds: `parent_action_id`, `causal_chain_id`, `scope_lease_id`, `input_taint_labels`, `memory_read_refs`, `pre_state_digest`, `run_context`, `signature.bundle`.

## Recipes (runnable examples)

| Recipe | Topic |
|---|---|
| recipe11–13 | rollout → SFT/DPO, compliance SFT |
| recipe14–16 | eval harness, significance, full loop |
| recipe17–18 | BFCL import, τ-bench import |
| recipe19–20 | τ-bench (alt), compliance training full pipeline |

Run any recipe: `python3 examples/recipeNN_*.py`

## Repository Boundaries

**This repository owns:**
- AEP evidence validation (`evomerge/validate/aep.py`)
- Evidence Admission Score and admission gate (`evomerge/validate/quality_gate.py`)
- Benchmark contamination statistics — dataset-level (paired McNemar, T0v2 triage, `eval_trust`)
- Calibrated trust score (`evomerge/trust_score.py`)
- Audit report generation (`evomerge/audit_report.py`)
- SFT / DPO / router record export (`evomerge/export/`)
- Benchmark adapters: BFCL, τ-bench, AgentHarm
- `eval_trust` paper companion

**Other repositories own — do not duplicate here:**

| Capability | Owner |
|---|---|
| AEP schema definition and versioning | `wasmagent-js` (`@wasmagent/aep`) |
| Runtime evidence collection (MCP firewall, AEP emitter) | `wasmagent-js` |
| Enterprise audit report, regulatory mapping | `open-agent-audit` |
| AgentBOM / MCP Posture / Trust Passport specifications | `agent-trust-infra` |
| Per-task contamination likelihood annotation (FAEP-level) | `fresharena` (issue #96) |
| Dynamic evaluation protocol | `fresharena` |

## Schema governance

Schemas in `schemas/` are the SSOT. Run `python scripts/export-schemas.py --check` to verify Pydantic models match. Nightly CI checks parity against wasmagent-js main.
