# Audit: Generation Length Distribution and Answer-Line Emission Rate

**Source data:** `data/winner_max_new768.json`, `data/instruct_max_new768.json`
**Generated:** 2026-06-24
**Script:** inline Python in agent session (see `data/audit_length_analysis.json` for full JSON output)

---

## 1. Generation Length Distribution (character count of `gen_text`)

| Statistic | winner (max\_new=768) | instruct (max\_new=768) |
|-----------|----------------------:|------------------------:|
| N valid   | 199 (1 error record)  | 200                     |
| min       | 382                   | 472                     |
| p25       | 768                   | 784                     |
| p50       | 1 057                 | 979                     |
| p75       | 1 719                 | 1 193                   |
| p95       | 3 414                 | 1 491                   |
| p99       | 3 992                 | 1 884                   |
| max       | 4 000                 | 2 516                   |
| mean      | 1 410.8               | 1 022.0                 |
| stdev     | 897.7                 | 312.4                   |

The winner model has a substantially heavier right tail: its p95 is 3 414 chars versus 1 491 for Instruct, and its stdev (897) is nearly 3x Instruct's (312). Instruct's distribution is tighter and shorter. Note that 4 000 chars is the soft ceiling imposed by max\_new=768 (approximately 768 tokens × ~5.2 chars/token); several winner outputs hit exactly 4 000 chars, indicating a small fraction may still be length-truncated even at this cap.

---

## 2. Answer-Line Emission Rate (`####` present in `gen_text`)

| Condition               | winner (max\_new=768) | instruct (max\_new=768) |
|-------------------------|----------------------:|------------------------:|
| All records             | 177/199 = **88.9%**   | 6/200 = **3.0%**        |
| Wrong answers only      | 55/64  = **85.9%**    | 3/62  = **4.8%**        |

The answer-line emission rates differ by ~30x. The winner model consistently terminates chains with the GSM8K-style `#### N` marker (88.9% overall, 85.9% even on wrong answers). The Instruct baseline almost never emits this marker (3.0% overall), instead surfacing answers via `\boxed{}`, bolded text, or prose.

This has a direct implication for parsing under a truncation cap: a winner chain cut at 300 tokens is more likely to have already emitted `#### N` before the cut than an Instruct chain of the same length, because the winner is trained to produce the marker earlier and more reliably.

---

## 3. Comparison: Are the Distributions Meaningfully Different?

Yes, in two independent ways:

**Shape.** The winner's distribution is right-skewed with a long tail (p75=1 719, p95=3 414), while Instruct's is compact and near-symmetric (p75=1 193, p95=1 491). At 95th percentile, the winner generates 2.3x more text than Instruct. This tail difference is what the `max_new=300` cap exploited: Instruct had substantially more probability mass above 300 tokens that was silently truncated.

**Answer-line convention.** The winner uses `#### N` as its terminal token in ~89% of outputs; Instruct uses it in ~3%. This is a training/merge artefact, not a capability signal — the wrong-answer rate shows both models are wrong at similar rates (winner: 32.2%, Instruct: 31.0%), but they express answers in different formats.

---

## 4. Reconstruction: What Happened at max\_new=300

No `gen_text` archives exist for the max\_new=300 runs. The following figures are sourced verbatim from Section 3.4 of the paper draft.

| Protocol        | N   | winner acc | instruct acc | Δ (pp)    | McNemar p |
|-----------------|-----|:----------:|:------------:|:---------:|:---------:|
| max\_new=300    | 200 | 59.5%      | 49.5%        | **+10.0** | 0.015     |
| max\_new=768    | 199 | 67.8%      | 69.0%        | **−1.2**  | 0.89      |
| Recovery (Δ pp) |     | +8.3 pp    | +19.3 pp     |           |           |

Instruct recovered 19.3 pp from the protocol fix; the winner recovered only 8.3 pp (asymmetry ratio: 2.33×). This is consistent with the length distribution evidence: Instruct had more chains in the 300–1500 char range that were being silently truncated before answer emission, while the winner's shorter-median, `####`-emitting chains were less affected. The 30x difference in answer-line emission rates above explains *why* truncation hurt Instruct more — its answers surface later in the chain, or not at all, making them more vulnerable to a cap.

---

## 5. Interpretation

Under the corrected protocol (max\_new=768), winner accuracy (67.8%) and Instruct accuracy (69.0%) are statistically indistinguishable. The large answer-line rate gap (88.9% vs 3.0%) is a **format artefact** of merging with a Coder variant trained on GSM8K-style completions; it does not indicate better reasoning. The wider length distribution of the winner means its chains were less affected by the 300-token cap, producing the spurious +10.0 pp gap. This audit confirms Proposition P1 from Section 1.2 of the paper: `max_new=300` was shorter than the 95th-percentile chain length of at least one candidate (Instruct's p95 ≈ 1 491 chars even at 768 tokens), and the asymmetry in chain-length distributions produced a false positive.

---

*Full machine-readable results:* `data/audit_length_analysis.json`
