# eval_trust self-test results

Synthetic-ground-truth verification, independent of the
case-study data. Run with `python benchmarks/self_test.py`.

## Scenarios

### NO_SIGNAL

- n per seed: 200
- p_a, p_b: 0.65, 0.65
- correlation rho: 0.5
- seeds: 20
- median p-value: 0.7535
- seeds reaching p < 0.05: 1 / 20
- expected (analytical): none should be significant

### REAL_SIGNAL

- n per seed: 400
- p_a, p_b: 0.65, 0.72
- correlation rho: 0.5
- seeds: 20
- median p-value: 0.0046
- seeds reaching p < 0.05: 17 / 20
- expected (analytical): most should be significant

## Wilson CI sanity

- 13/22 → 95% CI (38.7%, 76.7%) (paper Phase 11.5: 38.7%, 76.9%)
- 135/199 (case-study winner) → 95% CI (61.1%, 73.9%)
- 137/199 (case-study Instruct) → 95% CI (62.1%, 74.9%)

## Verdict

All assertions: **PASS**
