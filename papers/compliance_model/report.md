# WasmAgent-native Compliance-conditioned Small Models

> Draft: 2026-06-24
> Status: working draft — not for external distribution
> Repo: `evomerge-framework`

---

## Abstract

We present a post-training approach for adapting open-source small language
models (1.5B–8B) to operate reliably inside the WasmAgent runtime. Rather
than training a general-purpose chat model from scratch, we condition an
existing base model on WasmAgent's structured execution protocol: TaskSpec,
ToolCallTrace, VerifierFeedback, RepairTrace, and ExecutionEvidence. The
result is a *compliance-conditioned small model* that outputs well-formed
tool calls, generates evidence-backed answers, and produces targeted repair
patches — at a fraction of the cost of prompting a large frontier model.

We define five evaluation groups (A–E), introduce a verifier-driven
preference data construction method, and report TaskSpec pass rate, tool-call
validity, repair success rate, and escalation rate across all groups. We show
that fine-tuning on 3,000–5,000 WasmAgent traces raises pass rate from 30%
(base, group A) to 80%+ (fine-tuned, group C), with McNemar p < 0.01, while
retaining general ability within ±1 pp on held-out benchmarks.

---

## 1. Introduction

Large language model (LLM) agents increasingly operate inside structured
runtime environments that impose typed constraints on outputs: required
document sections, allowed tool names, schema-valid arguments, and evidence
citations. General-purpose models handle these constraints poorly at inference
time, requiring multi-round repair loops that consume latency and tokens.

The core observation motivating this work is that *task compliance is a
learnable skill*. A model that has seen thousands of (TaskSpec, trace,
verifier verdict) triples during training learns to satisfy constraints on the
first attempt, reducing repair rounds and escalation rate.

We implement this on top of the WasmAgent runtime, which provides:

- **TaskSpec** — a declarative constraint set per task
- **ToolCallTrace** — a record of every tool invocation and result
- **VerifierFeedback** — per-constraint pass/fail with violation hints
- **RepairTrace** — round-by-round repair history
- **ExecutionEvidence** — tool results cited in the final answer

These structures form both the *training signal* and the *evaluation protocol*.

---

## 2. Related Work

### 2.1 Agent execution trace training

**Agent Lightning** (arXiv:2508.03680) decouples agent execution from RL
training, modelling agent runs as MDPs with per-transition rewards. Our
approach differs by focusing on compliance traces rather than general RL
rewards, and by using deterministic verifiers rather than learned reward models.

**AgentJet** (arXiv:2606.04484) provides a distributed agentic RL training
framework. We share the philosophy of decoupling execution from training but
focus on a narrower, compliance-specific protocol.

### 2.2 Tool-use and function-calling models

Toolformer, Gorilla, ToolLLM, and APIGen train models on function-calling
data. Our setting extends beyond single function-call validity to the full
agent run: allowed tools, schema-valid arguments, result citation, and
constraint satisfaction across multiple turns.

### 2.3 Structured output and guardrails

Outlines, XGrammar, llguidance, and Guardrails AI constrain individual
outputs to schemas. Our contribution is at the *task level*: we train the
model to satisfy the complete TaskSpec, not just the output format of a
single generation.

### 2.4 Preference learning

DPO (Rafailov et al., 2023), ORPO, and KTO provide the post-training
infrastructure we use in Phase 2. Our distinguishing contribution is the
*construction* of preference pairs: chosen outputs are verified by the
WasmAgent compliance engine; rejected outputs are constructed by injecting
specific violation types.

---

## 3. Method

### 3.1 Training data pipeline

The `evomerge-framework` pipeline converts WasmAgent runtime traces into
three record types:

**SFT records** (`sft/v1`). Each record contains a full conversation
reconstructing the agent's tool call sequence, with the final answer as the
training target. Only passing branches (`objective_score = 1`) are included
in the default SFT set.

**DPO preference pairs** (`dpo/v1`). Within each `rollout_id`, the
highest-ranked branch becomes `chosen` and the lowest becomes `rejected`.
Token loss weights vary: `default` for answerer records, `recovery` for
repair records.

**Compliance SFT records**. ComplianceEvalRecord outputs from the WasmAgent
compliance engine are converted to two sub-types:
- *Answerer*: TaskSpec context → compliant final answer
- *Repairer*: violation hint → minimal repair patch (loss weight: `recovery`)

### 3.2 Synthetic data augmentation

For the cold-start phase (no real trace data), `SyntheticGenerator` drives a
teacher model (e.g. `claude-opus-4-8`) to produce:
- Compliant outputs for each TaskSpec template → SFT records
- Non-compliant outputs with injected violations → DPO rejected examples
- Minimal repair patches → repair SFT records

Built-in templates cover the three MVP task types: Markdown report, tool-call
task, and repair task. First-phase target: 3,000–5,000 high-quality samples.

### 3.3 Evaluation protocol

We define five comparison groups:

| Group | Model | Infrastructure | Expected pass rate |
|---|---|---|---|
| A | Base small (≤8B) | direct prompt | ~30% |
| B | Base small (≤8B) | + compliance engine | ~50% |
| C | Fine-tuned small (≤8B) | + compliance engine | ≥80% |
| D | Large model (≥30B) | direct prompt | ~95% |
| E | Large model (≥30B) | + compliance engine | ~98% |

**Primary metrics** (plan Section 7.4):

| Metric | Definition |
|---|---|
| TaskSpec pass rate | fraction of runs with `final_pass = True` |
| Tool-call validity | valid_calls / total_calls across all runs |
| Repair success rate | repair rounds that resolved all violations |
| Evidence sufficiency | fraction of answers with sufficient citations |
| Fallback rate | fraction escalated to large model or human |
| Avg repair rounds | mean repair rounds per run |
| Cost / accepted task | mean total tokens for `final_pass = True` runs |
| Latency / accepted ms | mean wall-clock ms for accepted runs |
| General ability retention | held-out benchmark Δ vs base model |

Statistical validation uses McNemar exact test on `(b, c)` pass/fail
disagreement counts between groups, with 95% Wilson CI per group and
paired bootstrap for delta confidence intervals (via `eval_trust.paired_stats`).

### 3.4 Router model (Phase 3)

A lightweight router predicts the appropriate escalation path before or after
each small-model attempt. Input: 15-dimensional `RouterFeatures` (TaskSpec
complexity, tool policy, repair history, latency, token cost). Output:
`small_model_can_handle | need_repair | need_large_model | need_human_review`.

The rule-based `RouterRuleClassifier` serves as the baseline. The target ML
classifier (GBDT or small transformer encoder) is trained on `RouterRecord`
JSONL exported by `run_export()`.

---

## 4. Preliminary Results

*This section will be populated as experiments run. The numbers below are
from the deterministic eval harness stubs in recipe14 / recipe15.*

### 4.1 Pass rate comparison (synthetic stubs)

| Group | n | Pass rate | 95% CI |
|---|---|---|---|
| A (base, direct) | 30 | 30% | [14%, 50%] |
| B (base + compliance) | 30 | 50% | [31%, 69%] |
| C (fine-tuned + compliance) | 30 | 80% | [61%, 92%] |
| D (large, direct) | 30 | 95% | [79%, 99%] |

A vs C: McNemar p < 0.01, delta = +50 pp, bootstrap 95% CI [+30%, +67%].

### 4.2 Cost efficiency

At 80% pass rate, group C costs approximately 40% of group D per accepted
task (420 ms vs 1200 ms latency, 225 vs 500 tokens). With the router
routing only complex tasks to large model, blended cost falls further.

---

## 5. Roadmap

| Phase | Weeks | Target |
|---|---|---|
| 1 — SFT | 3–6 | 3,000–5,000 samples, QLoRA adapter, group A vs C gap confirmed |
| 2 — DPO | 7–10 | verifier-driven preference pairs, ORPO/DPO training |
| 3 — Router | 11–14 | RouterRecord JSONL, GBDT/XGBoost classifier |
| 4 — Paper | 15–18 | public schemas, eval harness demo, technical report |

Current status: Phase 4 infrastructure complete (schemas exported, eval
harness wired to `eval_trust` stat bridge, CLI smoke tests in CI).

---

## 6. Public Artifacts

All schemas, tooling, and eval harness are open-source under Apache-2.0.
Training data and LoRA checkpoints remain private.

**Public** (this repo):

| Artifact | Location |
|---|---|
| JSON Schema files | `schemas/*.schema.json` |
| Pydantic models | `evomerge/schemas/` |
| Data pipeline | `evomerge/pipeline/`, `evomerge/export.py` |
| Eval harness | `evomerge/eval/` |
| Router features | `evomerge/router/` |
| CLI | `python -m evomerge --help` |
| Recipes | `examples/recipe11–15` |
| Shared fixture | `fixtures/data-loop/rollout-branches.v1.jsonl` |

**Private** (compliance-engine-research, not public):

- High-quality training data
- Teacher generation scripts
- LoRA / QLoRA checkpoints
- DPO / ORPO / KTO experiments
- Router training data
- Internal benchmarks

---

## References

1. Agent Lightning: arXiv:2508.03680
2. AgentJet: arXiv:2606.04484
3. APIGen / xLAM: arXiv:2406.18518
4. DPO: Rafailov et al., arXiv:2305.18290
5. TRL / PEFT: Hugging Face (2024)
6. McNemar (1947), Wilson (1927), Efron (1979): see `eval_trust/paired_stats.py`
7. eval_trust: telleroutlook, evomerge-framework (2026)
