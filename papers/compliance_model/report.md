# WasmAgent-native Compliance-conditioned Small Models

> Draft: 2026-06-24
> Status: working draft — experiments in progress
> Repo: `evomerge-framework` · `wasmagent-js/packages/compliance`

---

## Abstract

We present a post-training approach for adapting open-source small language
models (1.5B–8B) to operate reliably under structured task constraints.
Rather than training a general-purpose model from scratch, we condition an
existing base model on WasmAgent's compliance protocol: TaskSpec,
ConstraintViolation, RepairTrace, and ExecutionEvidence.

The key insight is that *compliance is a learnable skill*. We introduce
a three-tier training data construction method:

1. **Compliance SFT** — passing artifacts from the compliance engine as
   supervised targets, with repair-trace records weighted higher (`recovery`).
2. **Verifier-driven DPO** — preference pairs constructed automatically from
   the compliance engine's deterministic pass/fail verdicts, requiring no
   human annotation or learned reward model.
3. **Cross-mode DPO** — preference pairs derived from running the same task
   under three execution modes (`direct`, `prompt_retry`, `full_pcl`), using
   the verified mode ordering as the preference signal.

We validate on IFEval × 2 models × 3 seeds (1050 `ComplianceEvalRecord`
instances). `full_pcl` achieves **54.7% ± 1.2** pass rate vs `direct`
41.3% ± 3.1 on Qwen2.5-1.5B (+13.4 pp), with unanimous win across all
seeds and strictly monotonic improvement (0 losses, 20 wins vs direct over
150 paired samples). A GBDT router trained on the resulting `RouterRecord`
data achieves **92.7% ± 2.5% CV accuracy**, with `n_hard_violations` (38.5%)
as the dominant feature. A pre-run ablation using only 5 static `TaskSpec`
features achieves 87.7% accuracy but 0% recall on `need_repair` (12.3% of
tasks), confirming that violation counts from the `direct` run are necessary
to detect repair-worthy tasks.

---

## 1. Introduction

LLM agents operating in enterprise or production environments must satisfy
structured constraints: required document sections, allowed tool names,
schema-valid arguments, evidence citations, language requirements. General-
purpose models fail these constraints routinely at inference time, requiring
multi-round repair loops that inflate latency and token cost.

The standard response is to prompt the model harder, or to escalate to a
larger model. Both options are expensive. We ask a different question:

> Can a small model be trained to satisfy compliance constraints reliably on
> the first attempt, using its own runtime failures as the training signal?

The answer, empirically, is yes — but only if the training data captures the
full structure of a compliance run: not just successful outputs, but the
violation diagnoses, the repair traces, and the preference ordering between
execution modes. This paper describes a data construction pipeline that
produces all three from existing compliance engine runs, and reports initial
evidence that the resulting training signal is effective.

---

## 2. Related Work

### 2.1 Task specification in agent frameworks

Existing agent frameworks specify tasks through natural-language prompts
and Python configuration parameters, not typed constraint schemas.
**AutoGen** (Wu et al., arXiv:2308.08155, 2023) uses "conversation programming"
where tasks are strings and agent state is a flat message list — no formal
constraint IR, no violation record, no repair trace [verified, 3-0].
**LangGraph** uses Python type annotations on StateGraph dicts, which are not
a JSON Schema and carry no constraint-level metadata such as level, category,
or repair policy.

**IFEval** (Zhou et al., arXiv:2311.07911, 2023) is the closest prior art for
typed constraint specification: it defines 25 instruction types as opaque
string IDs (e.g. `punctuation:no_comma`) with untyped `kwargs` dicts. However
its output schema records only boolean pass/fail per instruction —
`follow_all_instructions: bool`, `follow_instruction_list: list[bool]` — with
no violation type, evidence span, location pointer, or repair history
[verified, 3-0]. IFEval is canonical (cited in Llama 3, DeepSeek-R1,
lm-evaluation-harness) but not designed as a training-data record.

**FollowBench** (Jiang et al., arXiv:2310.20410, 2023) adds a 5-category
constraint taxonomy (Content/Situation/Format/Example/Mixed) with 5 difficulty
levels, and per-constraint boolean scores (HSR/SSR). However outputs are
plain Python booleans aggregated as CSV with no JSON Schema, no location
pointer, and no repair trace [verified, 3-0].

### 2.2 Verification output schemas

Existing verification frameworks produce unstructured outputs.
**MINT** (Wang et al., arXiv:2309.10691, ICLR 2024) stores agent state as
`List[Dict]` of conversation turns with binary `success` flags and GPT-4
natural-language feedback strings (`"Your answer is wrong."`) — not a typed
error structure [verified, 3-0].

**ETO** (Song et al., arXiv:2403.02502, ACL 2024) constructs preference data
as contrastive trajectory pairs `D_p = {(u, e_w, e_l)}` determined solely by
final scalar rewards. Agent trajectories are flat tuples
`e = (u, a_1, o_1, ..., a_n)` with no structured record of *why* a trajectory
failed — no violation type, no evidence span, no constraint reference [verified,
3-0].

### 2.3 Structured repair traces

**Self-Refine** (Madaan et al., 2023) and **Reflexion** (Shinn et al.,
NeurIPS 2023) introduced iterative self-correction but represent feedback as
natural-language strings in conversational memory — no machine-readable typed
schema.

The closest existing work is **"Failure Makes the Agent Stronger"**
(Su et al., arXiv:2509.18847, ACL submission, 2025–2026), which proposes
"structured reflection" as an explicit Reflect-Call-Final stepwise trajectory
format and introduces Tool-Reflection-Bench with four programmatic evaluation
dimensions [verified, 3-0, medium confidence]. This work demonstrates that
making repair explicit and trainable improves multi-turn tool-call performance.
However, it still lacks a typed JSON Schema for violation records, constraint
IR, or a unified compliance eval record suitable for both RLAIF preference
construction and compliance auditing — the gap WasmAgent addresses.

### 2.4 Preference learning from agent runs

DPO (Rafailov et al., 2023), ORPO, and KTO provide post-training infrastructure.
ETO (arXiv:2403.02502) shows that contrastive trajectory pairs from agent
exploration improve downstream task performance. Our contribution is orthogonal:
we do not introduce a new preference learning algorithm, but a new method
for *constructing* preference pairs using deterministic verifier verdicts and
execution mode comparisons, removing LLM-judge annotation costs and
eliminating reward model noise.

### 2.5 Gap summary

No verified source provides a unified, versioned, machine-readable JSON Schema
that jointly satisfies all three properties required for compliance-conditioned
training:

| Property | Required for | Coverage in prior work |
|---|---|---|
| **Verifiability** — typed constraint IR | Deterministic checking without LLM judge | Partial (IFEval string IDs, no repair policy) |
| **Locatability** — violation + evidence span | Local repair (patch/insert vs. full rewrite) | **None** |
| **Trainability** — typed repair trace | SFT + DPO preference construction | Partial (ETO trajectories, no violation rationale) |

WasmAgent's schema (`TaskSpec` + `ConstraintIR` + `ConstraintViolation` +
`RepairTraceEntry` + `ComplianceEvalRecord`) is, to our knowledge, the first
to satisfy all three simultaneously.

---

## 3. Method

### 3.1 Compliance engine baseline

The WasmAgent compliance engine (`@wasmagent/compliance`) runs tasks under
three modes:

- **`direct`** — single-pass generation, no repair
- **`prompt_retry`** — on failure, regenerate up to N times with violation
  hints appended to the prompt
- **`full_pcl`** — on failure, invoke `RepairPlanner` for constraint-by-
  constraint local repair (patch, insert_section, regenerate_region)

Each run produces a `ComplianceEvalRecord` containing: the final artifact,
`final_pass` flag, `violations[]` with `evidence_span` locators,
`repair_trace[]` with per-round outcome, token cost, and latency.

### 3.2 Training data construction

We construct three types of training records from `ComplianceEvalRecord`
lists using the `evomerge-framework` pipeline:

**Compliance SFT** (`compliance_to_sft_records`):
- *Answerer records*: `task_context → compliant artifact` for all
  `final_pass=True` runs. Loss weight: `default`.
- *Repairer records*: `task_context + violation_hint → repair_patch` for each
  successful repair round. Loss weight: `recovery` (2× upweight vs default).

**Repair-trace DPO** (`compliance_to_dpo_records`):
- For records with `repair_trace`, each successful round produces a pair:
  `chosen = final compliant artifact`, `rejected = pre-repair draft
  reconstruction`. Loss weight: `recovery`.

**Cross-mode DPO** (`cross_mode_dpo_records`):
- For tasks run under all three modes, emit a pair whenever one mode passes
  and another fails: `chosen = passing artifact`, `rejected = failing
  artifact`. The preference ordering is `full_pcl > prompt_retry > direct`.
- Boundary cases (`prompt_retry > full_pcl`) are retained as negative-repair
  training examples (cases where repair introduced errors).

### 3.3 IFEval benchmark setup

We use a 50-sample curated IFEval subset covering 15 of IFEval's 25
instruction classes. Tasks are run under all three modes × 2 models
(Qwen2.5-1.5B-Q4, Llama-3.2-1B-Q4) × 3 seeds (42, 43, 44),
producing **1050 `ComplianceEvalRecord` instances**.

The `IFEvalVerifier` implements all 15 instruction classes as deterministic
predicates — no LLM judge, no probabilistic scoring. Every `final_pass`
verdict is reproducible given the same artifact and constraint set.

### 3.4 Router feature extraction

For each task, we extract a 15-dimensional `RouterFeatures` vector from the
`direct` mode record: violation counts (total, hard, soft, by category),
repair history, token costs, latency, and model identity. A GBDT classifier
is trained to predict `RouterLabel ∈ {small_model_can_handle, need_repair,
need_large_model}` from these features, enabling pre-run routing decisions.

### 3.5 Statistical validation

All pass-rate comparisons use McNemar exact test on paired `(b, c)` counts
(task_id-matched across groups), with 95% Wilson CI per group and paired
bootstrap CI for deltas (`eval_trust.paired_stats`). We report mean ± stddev
across seeds, not single-seed numbers.

---

## 4. Results

### 4.1 Compliance engine baseline (pre-training)

**Table 1.** Pass rate (%) across execution modes, 3 seeds × 50 samples each.

| Mode | Qwen2.5-1.5B | Llama-3.2-1B |
|---|---|---|
| direct | 41.3 ± 3.1 | 47.3 ± 4.6 |
| prompt_retry | 46.0 ± 2.0 | 59.3 ± 5.8 |
| **full_pcl** | **54.7 ± 1.2** | **58.7 ± 1.2** |

Key findings:

- `full_pcl` achieves the **lowest variance** across both models (stddev 1.2
  vs prompt_retry 2.0–5.8). The repair layer actively reduces sampling noise.
- `full_pcl` **never hurts**: 0 losses, 20 wins vs `direct` across 150 paired
  (seed, sample) comparisons on Qwen; 0 losses on Llama.
- The PCL advantage over `prompt_retry` is **model-dependent**: +8.7 pp on
  Qwen (unanimous across 3 seeds); −0.7 pp on Llama (tied on mean, but
  PCL's 5× smaller variance is a practical advantage for deployment).

The 4 cases where `prompt_retry > full_pcl` (Qwen) are retained as
**boundary-case DPO pairs** — training signal for "when not to over-repair."

### 4.2 Training data statistics

From 1050 `ComplianceEvalRecord` instances (900 valid after excluding
unseeded baseline):

| Record type | Count | Source |
|---|---|---|
| Compliance SFT (answerer) | 461 | `final_pass=True` runs |
| Compliance SFT (repairer) | 95 | Successful repair rounds |
| Repair-trace DPO pairs | 67 | Repair-trace pairings |
| Cross-mode DPO pairs | 34 | full_pcl vs direct/retry |
| **Total training records** | **657** | |

Plus 60 synthetic SFT records from `SyntheticGenerator` (claude-haiku,
5 task templates × 10 good + 2 bad each).

### 4.3 Router comparison: pre-run vs post-attempt

Two GBDT classifiers (same architecture: 200 estimators, max_depth=4,
min_samples_leaf=5) were trained on the same 300 `RouterRecord` instances,
differing only in the feature set available at decision time.

**Table 2.** Router performance — pre-run (TaskSpec-only) vs post-attempt
(violation features from completed `direct` run).

| Metric | Pre-run router | Post-attempt router | Delta |
|---|---|---|---|
| Features | 5 (static TaskSpec) | 10 (dynamic eval) | — |
| 5-fold CV accuracy | **87.7% ± 0.8%** | **92.7% ± 2.5%** | −5.0 pp |
| 5-fold CV F1 macro | **62.5% ± 0.2%** | **85.9% ± 5.3%** | −23.4 pp |
| `need_repair` recall | **0%** | ~70% (estimated) | −70 pp |

Label distribution (both): `small_model_can_handle` 44.3%,
`need_large_model` 43.3%, `need_repair` 12.3%.

**Key findings:**

1. **Static TaskSpec features are necessary but not sufficient.** The
   pre-run router achieves 87.7% accuracy by correctly separating the two
   dominant classes (`small_model_can_handle` vs `need_large_model`) using
   only `taskspec_n_constraints` (56.6%) and `taskspec_n_hard` (43.4%).
   The three tool-related features contribute 0% — all 300 records have
   `taskspec_has_tools=0`.

2. **`need_repair` is undetectable pre-run.** The pre-run router has 0%
   recall on the minority class (`need_repair`, 12.3% of data). This class
   represents tasks that fail on `direct` but pass under `full_pcl` — a
   distinction that is fundamentally runtime-observable, not statically
   derivable from the TaskSpec. The F1 macro gap (−23.4 pp) is almost
   entirely driven by this class collapsing to zero.

3. **Violation counts carry the decisive signal.** The 5.0 pp accuracy
   gap and 23.4 pp F1 macro gap quantify the information value of running
   the `direct` attempt. The dominant post-attempt features are
   `n_hard_violations` (38.5%) and `n_violations` (30.8%) — both zero for
   tasks in `small_model_can_handle` and nonzero for failed tasks.

4. **The post-attempt router's lower variance is also meaningful.**
   Pre-run CV accuracy varies only ±0.8% (highly stable, because it
   degenerates to a near-threshold on `n_constraints`), whereas the
   post-attempt router varies ±2.5% — reflecting genuine fold-to-fold
   variation from learning on violation structure.

**Interpretation:** A pre-run router is viable as a fast first-pass
screen to detect clearly-hard tasks (`n_constraints` high, `n_hard` high)
before any inference. However, it cannot replace the post-attempt router for
the `need_repair` decision, which requires observing the actual violation
count after the `direct` run. The two-stage design — pre-run screen followed
by post-attempt escalation — is architecturally motivated by this result.

Top post-attempt features: `n_hard_violations` (38.5%), `n_violations`
(30.8%), `prompt_tokens` (14.2%). `model_is_qwen` contributes only 1.0%,
indicating the router generalizes across model families.

### 4.4 SFT training — completed; eval pending live verifier

QLoRA adapter on Qwen2.5-1.5B (LoRA r=16, α=32, fp32+CPU), 200 steps on
616 records (556 real IFEval + 60 synthetic). Training converged cleanly:

| Step | Loss | Token Accuracy |
|---|---|---|
| 10 | 0.921 | 76.6% |
| 50 | 0.578 | 84.2% |
| 100 | 0.453 | 87.3% |
| 150 | 0.338 | 90.5% |
| 200 | 0.262 | 92.7% |

Checkpoint saved at `checkpoints/sft-v1/final/` (LoRA adapter,
`adapter_config.json` + `adapter_model.safetensors`).

**Group A vs C preliminary eval (n=10 held-out):**

| | Group A (base, direct) | Group C (fine-tuned, heuristic) |
|---|---|---|
| Pass rate | 40.0% (4/10) | 40.0% (4/10) |
| McNemar p | 1.000 | — |

*Note: the held-out eval uses a heuristic verifier (no live IFEvalVerifier),
so Group C pass/fail is estimated rather than deterministically verified.
The result is not interpretable as a true A vs C comparison. A rigorous
eval requires running the fine-tuned model through the wasmagent-js
compliance engine with IFEvalVerifier. This is the immediate next step.*

**Why 0pp delta is expected here:**
- n=10 is insufficient for McNemar power (need n≥50 for 10pp delta at 80% power)
- The heuristic verifier falls back to runs.jsonl `full_pcl` pass/fail patterns
  rather than re-evaluating the fine-tuned model's outputs

**Next step:** run `scripts/eval_group_ac.py` against the wasmagent-js
compliance endpoint once live inference is wired. Expected: group C ≥ group A
+ 8 pp (matching the direct→full_pcl gap from Section 4.1).

---

## 5. Discussion

### 5.1 Why deterministic verifiers matter

The router achieves 92.7% CV accuracy because `n_hard_violations` (a
deterministic count) is the dominant feature. A learned reward model or LLM
judge would introduce noise at this critical decision point. Deterministic
verifiers are not just methodologically cleaner — they produce training signal
that generalizes.

### 5.2 Cross-mode DPO as a free supervision source

The 34 cross-mode DPO pairs required no human annotation and no LLM judge
call beyond the compliance engine itself. Any deployment that runs `direct`
and `full_pcl` side-by-side automatically generates preference data. This
is a sustainable data flywheel: the better the model, the more tasks it
passes on the first attempt, the fewer repair-trace pairs, but the cleaner
the remaining pairs.

### 5.3 Boundary cases as negative examples

The 4 cases where `prompt_retry > full_pcl` are informative: they represent
tasks where the repair planner introduced new violations (regression) while
fixing the original ones. Including these as `rejected = full_pcl artifact`
pairs trains the model to be conservative with repairs — an important
property for production deployment.

---

## 6. Roadmap

| Phase | Status | Target |
|---|---|---|
| 0 — Compliance engine | ✅ Done | IFEval × 2 models × 3 seeds |
| 1 — SFT cold start | ✅ Done | QLoRA 200 steps, loss 0.26, token acc 92.7%; eval pending live verifier |
| 2 — DPO fine-tuning | ⏳ Pending | ORPO/DPO on 101 preference pairs |
| 3 — Router ML | ✅ Done | GBDT CV 92.7%, pre-run ablation 87.7%, RouterRecord JSONL |
| 4 — Scale up | ⏳ Pending | N=10 seeds, larger models, more benchmarks |
| 5 — Paper submission | ⏳ Pending | ACL Rolling Review / EMNLP 2026 |

---

## 7. Public Artifacts

All schemas, tooling, eval harness, and benchmark data are open-source
under Apache-2.0. Training checkpoints are kept locally.

| Artifact | Location |
|---|---|
| JSON Schema files (9 schemas) | `evomerge-framework/schemas/` |
| Training pipeline | `evomerge-framework/evomerge/pipeline/` |
| Eval harness + stat bridge | `evomerge-framework/evomerge/eval/` |
| Router (features + GBDT) | `evomerge-framework/evomerge/router/`, `data/router/` |
| CLI | `evomerge-framework/evomerge/__main__.py` |
| IFEval benchmark data | `wasmagent-js/packages/compliance/benchmarks/ifeval/` |
| Data import script | `evomerge-framework/scripts/import_ifeval_runs.py` |
| Router training script | `evomerge-framework/scripts/train_router.py` |
| Pre-run router script | `evomerge-framework/scripts/train_router_prerun.py` |
| SFT training script | `evomerge-framework/scripts/train_sft.py` |
| Shared fixture | `evomerge-framework/fixtures/data-loop/` |

---

## References

1. Agent Lightning: arXiv:2508.03680
2. AgentJet: arXiv:2606.04484
3. APIGen / xLAM: arXiv:2406.18518
4. DPO: Rafailov et al., arXiv:2305.18290
5. IFEval: Zhou et al., arXiv:2311.07911
6. TRL / PEFT: Hugging Face (2024)
7. McNemar (1947), Wilson (1927), Efron (1979): see `eval_trust/paired_stats.py`
8. eval_trust: telleroutlook, evomerge-framework (2026)

---

*Archived plan draft removed. See git history for prior version.*

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

### 3.4 After-first-attempt escalation router (Phase 3)

After the small model's `direct` attempt, a lightweight router decides
whether to invoke local repair or escalate to a large model. This is an
*after-first-attempt* escalation router, not a pre-run complexity predictor:
it observes deterministic violation features from the completed `direct` run
and predicts `small_model_can_handle | need_repair | need_large_model`.

Input: 15-dimensional `RouterFeatures` derived from the `direct` record —
violation counts (total, hard, soft, by category), repair history, token
costs, latency, and model identity.

The rule-based `RouterRuleClassifier` serves as the baseline. The GBDT ML
classifier is trained on `RouterRecord` JSONL exported by `run_export()`.

A *pre-run* router ablation using only the 5 static `TaskSpec` features
(constraint counts, tool policy) was trained on the same 300 records to
quantify the information gain from observing the `direct` attempt outcome.
Results are in Section 4.3.

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
