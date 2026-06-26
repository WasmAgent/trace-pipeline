# Appendix

This appendix provides reproducibility information, key numeric results, and environment details for the eval-trust paper.

## A  Reproducibility

### A.1  Regenerating figures

All figures are generated from data files committed in this repository. No external project state is required.

```bash
# From repo root
python papers/eval_trust/scripts/make_figures.py
# Outputs: papers/eval_trust/figures/{fig1,fig2,fig3}.{pdf,png}
```

### A.2  Regenerating this appendix

```bash
python papers/eval_trust/scripts/generate_appendix.py \
    --output papers/eval_trust/APPENDIX.md
```

### A.3  Compliance benchmark sweep (IFEval)

Reproduces the IFEval × Qwen2.5-1.5B-Q4 results in Table B.1 below.

```bash
# From wasmagent-js repo root — runs 3 seeds × 50 samples
bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=42
bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=43
bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=44

# Aggregate across seeds
bun packages/compliance/benchmarks/ifeval/compare-seeds.ts
```

### A.4  Data file checksums

SHA-256 digests of all tracked data files at time of appendix generation.

| File | Size | SHA-256 |
|------|------|---------|
| `data/audit_length_analysis.json` | 2 KB | `4126ac839a5d37a7...` |
| `data/t0v2_aggregate.json` | 43 KB | `521f22f6b39995dd...` |
| `data/self_consistency_full.json` | 23 KB | `0b595ad926caaf6e...` |
| `data/instruct_max_new768.json` | 235 KB | `ecbde400d54b2d36...` |
| `data/winner_max_new768.json` | 347 KB | `995a940647aad998...` |
| `data/gsm8k_dev_200.json` | 125 KB | `d2fcdbbe7664b1a6...` |
| `data/quantization_granularity/summary.json` | 2 KB | `e41ab6a8f3044a89...` |
| `data/synthetic_4algo/marginal_history_anonymized.json` | 4 KB | `b7089fec340133ba...` |

## B  Key Results

### B.1  IFEval pass-rate summary

Results from the Compliance Engine PCL evaluation (3 seeds × 50 samples per condition).

| Model | Benchmark | Method | Pass-rate | Std | Notes |
|-------|-----------|--------|-----------|-----|-------|
| Qwen2.5-1.5B-Q4 | IFEval | `full_pcl` | 54.7% | ±1.2% | 3 seeds × 50 samples |
| Qwen2.5-1.5B-Q4 | IFEval | `prompt_retry` | 46.0% | ±2.0% | baseline |
| Llama-3.2-1B | IFEval | `full_pcl` | ties prompt_retry | 5× smaller variance | PCL reduces variance |

**Headline finding**: on IFEval × Qwen2.5-1.5B-Q4, `full_pcl` achieves **54.7% ± 1.2%** vs `prompt_retry` **46.0% ± 2.0%** (**+8.7 pp**, 3 seeds × 50 samples).

On Llama-3.2-1B, PCL ties `prompt_retry` on mean but has 5× smaller variance, suggesting PCL primarily reduces stochastic failures on harder models.

### B.2  Result data location

| Description | Path (wasmagent-js repo) |
|-------------|-------------------------|
| Raw records (1 050 entries) | `packages/compliance/benchmarks/ifeval/results*/` |
| Multi-seed phase reports | `packages/compliance/benchmarks/ifeval/results-multi-seed*/*.md` |
| Cross-model summary | `packages/compliance/benchmarks/ifeval/results-multi-seed-llama/CROSS-MODEL-2026-06-24.md` |

## C  Environment

### C.1  Runtime

| Component | Version / value |
|-----------|-----------------|
| Runtime | Bun ≥ 1.2 |
| Language | TypeScript 5 |
| Package manager | npm workspaces + turbo |
| Test runner | `bun test` |
| Linter | Biome |

### C.2  Model environment

| Parameter | Value |
|-----------|-------|
| Qwen2.5-1.5B quantization | Q4 (GGUF, per-tensor) |
| Llama-3.2-1B quantization | Q4 (GGUF, per-tensor) |
| IFEval instruction classes | 15 |
| Compliance verifier built-in checks | 7 |
| Repair strategies | PatchStrategy, InsertSectionStrategy, RegenerateRegionStrategy |

### C.3  Relevant packages

| Package | Stability |
|---------|-----------|
| `@wasmagent/compliance` | Alpha (schema versioned) |
| `@wasmagent/core` | Stable |
| `@wasmagent/kernel-quickjs` | Stable |
| `@wasmagent/evals-runner` | Growth |

