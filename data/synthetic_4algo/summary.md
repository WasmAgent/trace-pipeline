# R2.2 Synthetic algorithm comparison
Setup: 4 TVs, shape (256, 256), 10% sparse signal + noise.

## RMS strength (||merged - base||) — higher = stronger merge

| λ | DELLA | Breadcrumbs | SCE | TIES | naive |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.0353 | 0.0092 | 0.0352 | 0.0188 | 0.0087 |
| 0.3 | 0.1058 | 0.0277 | 0.1057 | 0.0563 | 0.0261 |
| 0.5 | 0.1764 | 0.0462 | 0.1762 | 0.0939 | 0.0436 |
| 0.7 | 0.2469 | 0.0647 | 0.2466 | 0.1315 | 0.0610 |
| 1.0 | 0.3527 | 0.0924 | 0.3523 | 0.1878 | 0.0871 |

## Sign-flip % (vs base) — TIES cliff: spike at high λ

| λ | DELLA | Breadcrumbs | SCE | TIES |
|---|---:|---:|---:|---:|
| 0.1 | 6.48% | 1.82% | 6.44% | 3.55% |
| 0.3 | 13.28% | 4.99% | 13.16% | 8.62% |
| 0.5 | 16.55% | 7.67% | 16.31% | 11.87% |
| 0.7 | 18.64% | 9.81% | 18.38% | 14.08% |
| 1.0 | 20.95% | 12.15% | 20.54% | 16.42% |

## Abs max (largest single weight after merge) — explosion indicator

| λ | DELLA | Breadcrumbs | SCE | TIES | naive |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.498 | 0.485 | 0.498 | 0.472 | 0.464 |
| 0.3 | 0.924 | 0.544 | 0.924 | 0.676 | 0.482 |
| 0.5 | 1.506 | 0.623 | 1.506 | 1.065 | 0.524 |
| 0.7 | 2.087 | 0.771 | 2.087 | 1.462 | 0.612 |
| 1.0 | 2.960 | 0.993 | 2.960 | 2.058 | 0.779 |

## Interpretation (data-driven, replacing pre-experiment hypothesis)

We hypothesised before running this bench that TIES would show "cliff
collapse" with the highest abs_max and most aggressive sign flips at high
λ — matching Phase 9.4's real-Qwen observation. The synthetic data
*partially* confirms this and *partially* contradicts it. Worth recording
both because the contradiction is the more interesting finding.

### What the data actually shows (λ = 1.0)

| Algorithm | RMS strength | Sign flip % | abs_max | Description |
|---|---:|---:|---:|---|
| Breadcrumbs | 0.092 | 12.2% | **0.993** | Most conservative on every metric |
| TIES        | 0.188 | 16.4% | 2.058 | Mid-aggressive |
| SCE         | 0.352 | 20.5% | 2.960 | Most aggressive |
| DELLA       | 0.353 | 21.0% | 2.960 | Most aggressive |
| naive_linear | 0.087 | 11.6% | 0.779 | Reference |

### What this tells us

- **Breadcrumbs is the only "soft" algorithm by every metric.** Its
  two-tail trim explicitly removes the largest-magnitude entries, which
  caps abs_max well below the others. Its sign-flip rate is the closest
  to naive linear. If you need a merge to "not break things at high λ",
  Breadcrumbs is the structural choice.

- **DELLA and SCE are *more* aggressive than TIES on this synthetic.**
  DELLA's Bernoulli sampling at density=0.7 keeps roughly 70% of entries,
  more than TIES at trim_ratio=0.2 (which keeps 80% but then sign-elects
  a single sign per parameter). SCE at density=0.5 keeps top-50%
  per-tensor. The amplification from λ then hits more entries in DELLA/SCE
  than in TIES.

- **TIES "cliff collapse" is not visible in synthetic data.** All four
  curves (sign-flip vs λ, abs_max vs λ) are smooth-monotonic. The Phase
  9.4 cliff was probably tied to *specific structural properties of real
  Qwen weights* (e.g., certain layers having near-zero-magnitude entries
  whose sign election is unstable) rather than a generic property of TIES.
  Real-Qwen R2.2 (deferred to scripts/r2_2_qwen_bench.py) is needed to
  characterise that.

### Calibration takeaway

Hyperparameters used here were **textbook defaults**, not project-tuned:
- DELLA density=0.7 (paper §3.2 default)
- Breadcrumbs alpha=0.05, beta=0.10 (paper §4.1 default)
- SCE density=0.5 (mergekit default)
- TIES trim_ratio=0.2 (paper default)

Direct comparison at default settings is therefore **not** a comparison
of the algorithms' best-case behaviour — it's a comparison of what each
out-of-the-box configuration does. Per-algorithm hyperparameter tuning
(R2.2 Qwen bench, R3) will produce a fair comparison; this synthetic
bench is a *data-flow sanity check*, no more.

### Implications for the paper

The R4a paper's §5.2 quantization granularity finding (per-tensor int4
collapses to 0%, group-32 int4 recovers to 63%) has a **structural**
analog here: at the same nominal "lambda", what an algorithm actually does
is governed by a primitive (granularity / sampling / density / trim) that
isn't visible from the headline parameter. This synthetic bench is an
example of that pattern reaching the merging side, not just the
quantization side.

We will not promote any of these synthetic numbers to the paper; only
real-model results from R2.2 Qwen bench will go there.

Synthetic data is no substitute for real-model evaluation — see
scripts/r2_2_qwen_bench.py for the GSM8K case study (W4-W7).
