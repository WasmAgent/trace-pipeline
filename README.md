# trace-pipeline

[![CI](https://github.com/WasmAgent/trace-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/WasmAgent/trace-pipeline/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red.svg)](papers/eval_trust/draft.pdf)

**Measurement trust and trace-to-training backend for WasmAgent.**

> *trace-pipeline is the research backend that tells you whether your evaluation is trustworthy and whether your runtime traces are safe enough to become training data.*

This repository is the third layer of the [WasmAgent Trustworthy Agent Training Loop](docs/ecosystem-map.md):

```
wasmagent-js  →  bscode  →  trace-pipeline  →  better runtime
(runtime)        (workload)  (measurement + data)    (fed back)
```

It has two modules:

1. **`eval_trust`** — benchmark audit: paired statistics, contamination checks, T0v2 triage.
   Companion to *"Silent Contamination in LLM Merging Evaluation"* ([paper](papers/eval_trust/draft.pdf)).
2. **`evomerge`** — trace-to-training pipeline: `ComplianceEvalRecord` → SFT / DPO / router records.

---

## Why this exists

We spent **5 months** chasing a **+10 pp** GSM8K improvement on a Qwen2.5-1.5B
merge that survived three rounds of verification, paired McNemar
significance (p = 0.015), and multi-seed runs. Then we noticed
`max_new_tokens` was hardcoded to 300 in our generation runner.

Re-ran with `max_new_tokens=768`. The +10 pp **collapsed to −1.0 pp**
(p = 0.89). The "improvement" was the baseline being silently truncated
more often than the merge candidate.

This toolkit is the audit we wish we had run on day 1.

📄 **Read the paper:** [`papers/eval_trust/draft.pdf`](papers/eval_trust/draft.pdf)
(30 pages, 3 figures, 23 references — tells the full story).

---

## 30-second demo: reproduce the case-study flip

```bash
git clone https://github.com/WasmAgent/trace-pipeline
cd trace-pipeline
pip install scipy
python - <<'PY'
import json
from scipy.stats import binomtest

# Real raw logs from the case study (n=199, both runs evaluated greedy on
# the same GSM8K dev items, only max_new_tokens differs):
with open("data/winner_max_new768.json") as f:
    winner = json.load(f)
with open("data/instruct_max_new768.json") as f:
    instruct = json.load(f)

wm = {r["id"]: r["correct"] for r in winner["results"] if "correct" in r}
im = {r["id"]: r["correct"] for r in instruct["results"] if "correct" in r}
common = sorted(set(wm) & set(im))
b = sum(1 for i in common if im[i] and not wm[i])  # Instruct-only correct
c = sum(1 for i in common if not im[i] and wm[i])  # winner-only correct
p = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue

print(f"AUDITED: n={len(common)}, b={b}, c={c}, p={p:.4f}")
# AUDITED: n=199, b=29, c=27, p=0.8939   ← the +10 pp wasn't real

# For comparison, here's the same paired test on the BROKEN protocol
# (max_new=300, the original Phase 13 numbers):
print("ORIGINAL: n=200, b=21, c=41, p=0.0151   ← was paired-significant")
PY
```

Output:

```
AUDITED:  n=199, b=29, c=27, p=0.8939   ← the +10 pp wasn't real
ORIGINAL: n=200, b=21, c=41, p=0.0151   ← was paired-significant
```

That's the entire case study, on real data, in one screenful.

---

## What's in the toolkit

📖 **[EXAMPLES.md](EXAMPLES.md)** — 17 copy-pasteable recipes (10 eval_trust + 7 evomerge pipeline, including recipe17 AEP end-to-end demo).
🐍 **[`examples/`](examples/)** — same 17 as standalone runnable .py.
🛠️ **[`Makefile`](Makefile)** — dev shortcuts (`make help`).
🧪 **[`benchmarks/`](benchmarks/)** — synthetic ground-truth self-test.
📄 **[`papers/eval_trust/`](papers/eval_trust/)** — paper PDF + LaTeX + figures + scripts.
📄 **[`papers/compliance_model/`](papers/compliance_model/)** — compliance training report + schema novelty analysis.

```
eval_trust/   paired_stats / conformal_ci / lm_eval_bridge / t0v2 / exploit_surface
evomerge/     schemas / pipeline / adp / rl / capability / context_compile / security /
              validate / benchmarks / router / trust_score / registry / audit_report / provenance
data/         case-study logs + synthetic + quantization summary
tests/        441 tests (53 eval_trust + 388 evomerge pipeline)
```

**eval_trust**: Pure Python + NumPy + (optional) SciPy. No GPU. No model inference.
**evomerge**: Pydantic v2 + scikit-learn (optional). Converts compliance traces to training data.

---

## Three reasons to use it

**1. Cheap.** Audit cost ~milliseconds on 200 items. A `pytest` run
takes 0.04 s. The case study took 4 hours of laptop time end-to-end.

**2. Falsifiable.** Every channel is a deterministic predicate on
`(question, expected, gen_text)`. No probabilistic classifier, no
LLM-as-judge, no opaque model — you can replicate every label by hand.

**3. Embarrassing if you don't.** A small-delta merging claim on a
standard benchmark, without a paired McNemar `(b, c)` count and a
SC-5 lottery rate report, is consistent with a real improvement, with
a primitive contamination, and with several mechanisms in between. The
audit costs you nothing and removes the ambiguity. Reviewers will
increasingly expect it.

---

## Tests

```bash
pip install scipy pytest
PYTHONPATH=. pytest tests/ -q
```

Or via the project Makefile (`make help` shows all 14 targets):

```bash
make install    # pip install -e ".[dev]"
make pytest     # 226 unit + integration tests
make ci         # everything CI runs (test + lint + reproducer + self-test + examples)
make paper      # rebuild draft.pdf + arxiv_upload.tar.gz (needs pandoc + tectonic)
```

---

## WasmAgent trace-to-training pipeline (`evomerge` package)

This repository also contains a second package — `evomerge` — that converts
WasmAgent runtime traces into training data for compliance-conditioned small
model post-training.

### Install

```bash
pip install -e ".[dev]"
```

### 30-second demo: fixture → SFT + DPO

```bash
python -m evomerge export \
  --rollout fixtures/data-loop/rollout-branches.v1.jsonl \
  --out-dir /tmp/demo
cat /tmp/demo/manifest.json
# {"n_sft": 1, "n_dpo": 1, "n_ppo": 2, ...}
```

### Package layout

```
evomerge/
├── schemas/        Pydantic models (RolloutBranchRecord, ComplianceEvalRecord,
│                   SftTrainingRecord, DpoTrainingRecord, PpoTrainingRecord …)
├── pipeline/       trace → SFT / DPO / PPO / compliance-SFT converters
├── io.py           load_jsonl / write_jsonl / load_rollouts / load_router_records
├── export.py       run_export() — full pipeline in one call
├── validate/       contamination (8-gram Jaccard) + schema structural checks
├── synthesize/     TaskSpec templates + SyntheticGenerator (teacher model)
├── eval/           EvalHarness (A/B/C/D/E groups), EvalMetrics, stat_bridge
├── router/         RouterFeatures, RouterLabel, RouterRuleClassifier
└── __main__.py     CLI: export / adp-export / rl-export / compile-context / router / validate / synthesize
```

### CLI

| Command | What it does |
|---|---|
| `python -m evomerge export` | rollout + compliance JSONL → sft/dpo/ppo/router.jsonl |
| `python -m evomerge adp-export` | rollout-wire/v1 → ADP (Agent Data Protocol) episode JSONL |
| `python -m evomerge rl-export` | rollout-wire/v1 → RL transition records (build/policy/cost reward) |
| `python -m evomerge compile-context` | rollout traces → long-context QA or router/critic records |
| `python -m evomerge validate` | schema + contamination check on any training JSONL |
| `python -m evomerge router` | batch routing predictions with rule classifier |
| `python -m evomerge synthesize` | generate synthetic samples via teacher model |
| `python -m evomerge validate-aep` | validate AEP (Agent Evidence Protocol) records |
| `python -m evomerge lint-benchmark` | anti-reward-hacking exploit surface check on task directory |
| `python -m evomerge trust-score` | compute AgentTrustScore (9-dim geometric mean) |
| `python -m evomerge audit-report` | generate full Markdown benchmark audit report |
| `python -m evomerge receipt` | produce SCITT-style run provenance receipt |
| `python -m evomerge registry-register` | register artifact in Evidence Registry |
| `python -m evomerge registry-list` | list Evidence Registry entries by type |
| `python -m evomerge import-bfcl` | BFCL v4 JSONL → rollout-wire JSONL |
| `python -m evomerge import-mcp-atlas` | MCP-Atlas JSONL → rollout/AEP |
| `python -m evomerge import-oai-agents` | OpenAI Agents SDK trace → AEP JSONL |
| `python -m evomerge import-langsmith` | LangSmith/LangGraph trace → AEP JSONL |
| `python -m evomerge import-ms-agent-framework` | Microsoft Agent Framework trace → AEP JSONL |
| `python -m evomerge import-adk` | Google ADK trace → AEP JSONL |
| `python -m evomerge import-a2a-task` | A2A task → AEP JSONL |
| `python -m evomerge import-terminal-bench` | Terminal-Bench → rollout/AEP JSONL |
| `python -m evomerge import-tau-bench` | τ³-bench → rollout/AEP JSONL |
| `python -m evomerge import-tool-sandbox` | ToolSandbox → AEP JSONL |
| `python -m evomerge import-agent-harm` | AgentHarm/OS-Harm/CUAHarm → AEP JSONL |
| `python -m evomerge import-otel` | OTel spans JSONL → AEP JSONL |

### New modules (since v0.2)

| Module | Location | Purpose |
|---|---|---|
| ADP export | `evomerge/adp/export.py` | rollout → Agent Data Protocol episodes |
| RL transitions | `evomerge/rl/export.py` | rollout → (state, action, reward, done) tuples |
| Capability taxonomy | `evomerge/capability/taxonomy.py` | 11-tag deterministic step tagger |
| Capability attribution | `evomerge/capability/attribution.py` | mine failure modes from pass/fail branch pairs |
| Trace-to-context | `evomerge/context_compile/compiler.py` | long-context QA + router/critic records |
| MCP security eval | `evomerge/security/mcp.py` | McpSecurityEvalRecord schema for firewall events |
| Dataset card | `scripts/generate-dataset-card.py` | auto-generate DATASET_CARD.md from manifest.json |
| AEP validation | `evomerge/validate/aep.py` | validate AEP records + evidence completeness |
| Benchmark linter | `evomerge/security/benchmark_linter.py` | 6-check exploit surface scanner |
| Run provenance | `evomerge/provenance.py` | SCITT-style RunReceipt + RunReceiptBuilder |
| Evidence Registry | `evomerge/registry.py` | signed-JSON artifact registry (8 entry types) |
| Agent Trust Score | `evomerge/trust_score.py` | 9-dim geometric mean trustworthiness score |
| Audit report | `evomerge/audit_report.py` | AEP + linter + provenance → Markdown report |
| AEP→router bridge | `evomerge/router/aep_bridge.py` | features_from_aep() + label_from_aep() |
| Trace Card | `evomerge/dataset_card.py` | TraceCard + generate_trace_card() |
| Exploit taxonomy | `eval_trust/exploit_surface.py` | 6-surface agent benchmark exploit taxonomy |
| Benchmark adapters | `evomerge/benchmarks/` | BFCL v4, MCP-Atlas, Terminal-Bench, τ-bench, ToolSandbox, AgentHarm, OAI, LangSmith, MS-AF, ADK, A2A |

### Schema contract

Schemas mirror wasmagent-js TypeScript interfaces and are validated by CI:

```bash
python scripts/check-schema-fields.py
# [OK] rollout-wire (contract)
# [OK] training-record/dpo (contract)
# ✓ no schema drift detected
```

Pass `--wasmagent-js /path/to/repo` to check against the canonical JSON Schema files.

### Shared fixture

`fixtures/data-loop/rollout-branches.v1.jsonl` — 2-branch rollout fixture
(1 pass / 1 fail). Must be byte-identical across trace-pipeline, wasmagent-js,
and bscode. Sync all three repos in the same PR when changing.

### Recipes

| File | Topic |
|---|---|
| `examples/recipe11_rollout_to_sft.py` | rollout JSONL → SFT records |
| `examples/recipe12_rollout_to_dpo.py` | rollout JSONL → DPO preference pairs |
| `examples/recipe13_compliance_sft.py` | ComplianceEvalRecord → answerer + repairer |
| `examples/recipe14_eval_harness.py` | A/B/C/D/E comparison harness |
| `examples/recipe15_significance.py` | McNemar + bootstrap: C > A at p < 0.05 |

---

## Citation

```bibtex
@misc{evaltrust2026,
  title  = {Silent Contamination in {LLM} Merging Evaluation:
            A Case Study from a 5-Month Misadventure},
  author = {{telleroutlook}},
  year   = {2026},
  url    = {https://github.com/WasmAgent/trace-pipeline},
}
```

## License

- **Code** (`eval_trust/`, `evomerge/`, `tests/`): Apache-2.0 (see `LICENSE`)
- **Paper** (`papers/eval_trust/draft.{pdf,md}`): CC BY 4.0
- **Data** (`data/`): CC BY 4.0 (these are evaluation logs of public Qwen
  models on GSM8K, both of which are publicly licensed)

---

## Status

Pre-arxiv preprint. The paper is camera-ready (PDF compiles cleanly via
pandoc + tectonic). Awaiting an arxiv endorsement to assign an arXiv ID.

If you'd like to **endorse** this submission for arxiv `cs.CL`, please open
an issue or DM — endorsement is a 30-second click, no paper review needed.
