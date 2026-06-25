# Contributing to eval_trust

Thanks for your interest. This is a small, focused toolkit. Contributions
that align with the project's scope are very welcome; please read this
short guide first.

## Scope

`eval_trust` is the audit-side of LLM-merging evaluation:

- ✅ Paired statistics, conformal CI, T0v2 channels, SC lottery measurement
- ✅ Adapters that consume outputs of standard runners (lm-evaluation-harness)
- ✅ Reproducer scripts that exercise the toolkit on real data
- ❌ New benchmarks (use lm-eval-harness instead)
- ❌ New merging algorithms (the paper is methodology, not algorithms)
- ❌ Heavy GPU dependencies (the toolkit must run on a laptop in seconds)

If you're unsure whether your idea is in scope, please open an issue
to discuss before writing code.

## Quick start

```bash
git clone https://github.com/WasmAgent/trace-pipeline
cd trace-pipeline
pip install -e ".[dev]"

# Run all tests
PYTHONPATH=. pytest tests/ -q

# Run the case-study reproducer (sanity-check the install)
python run_audit.py
```

## Pull request checklist

Before submitting:

- [ ] **Tests pass.** `PYTHONPATH=. pytest tests/ -q` is green.
- [ ] **New code has tests.** Aim for one test per public function.
      Edge cases (empty input, n=1, off-by-one) explicitly tested.
- [ ] **Lint passes.** `ruff check eval_trust/ tests/ run_audit.py` is clean.
- [ ] **Reproducer still works.** `python run_audit.py` produces the
      expected case-study verdict.
- [ ] **Docstrings on public APIs.** Every public function has a one-line
      summary + Args + Returns. Math is OK in docstrings (we render at
      paper level).
- [ ] **English only.** Code, docstrings, comments, commit messages,
      and docs are all in English.
- [ ] **Honest commit messages.** If your PR is a tradeoff, say so;
      if it's a bugfix, name the bug; don't over-claim.

## What we won't merge

- Code that depends on a GPU at import time (acceptable: optional
  imports inside functions).
- Code that depends on a specific model checkpoint or a closed-source
  dataset.
- Changes to the `papers/` directory that alter cited numbers without
  updating `papers/eval_trust/numbers_cross_check.json` to keep them in
  sync. The paper-source-of-truth is the published arxiv version; this
  repo is the reproducibility companion.
- Reformat-only PRs ("ran black on everything"). The existing style
  is intentional; if you find a real ruff violation, point it out in
  an issue first.

## Reporting bugs

Please use the GitHub issue templates:

- **Numeric mismatch** — if `run_audit.py` or `make_figures.py`
  produces numbers that disagree with what the paper claims, that's
  a high-priority bug. Include the OS / Python / NumPy versions and
  the exact diff.

- **API behaviour issue** — if a function in `eval_trust/` returns
  something surprising or crashes on input that should be valid,
  open an issue with a minimal reproducer.

- **Documentation issue** — if EXAMPLES.md or the paper has an error,
  point to the line and explain what's wrong.

For *security* issues (e.g. arbitrary code execution from a malformed
input file), please email rather than file a public issue. We don't
expect any in this codebase, but the channel is open.

## License of contributions

By submitting a pull request, you agree that your contribution will be
licensed under the same Apache-2.0 license as the rest of the code.
The paper text is CC BY 4.0; data files are CC BY 4.0.

## Style

- Python: PEP 8, enforced by `ruff`. Line length 100. Type hints on
  public APIs.
- Imports: stdlib, then third-party, then local.
- Tests: pytest classes for grouping, descriptive method names. Each
  test asserts one thing.
- Commit messages: imperative mood ("Add foo", not "Added foo"),
  ≤ 72 chars on the first line, blank line, then a body if needed.

## Questions

For methodology questions about the paper, open a GitHub issue (we
prefer public discussion). For private questions, reach out via the
contact in `CITATION.cff`.
