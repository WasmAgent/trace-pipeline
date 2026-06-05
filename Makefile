# Makefile — common dev tasks for evomerge-framework
#
# Usage:
#   make help        # show all targets
#   make test        # run pytest + reproducer + self-test + 10 examples
#   make lint        # ruff check
#   make figures     # regenerate paper figures from data/
#   make paper       # rebuild draft.pdf + arxiv_upload.tar.gz
#   make all         # test + lint + figures + paper
#   make clean       # remove generated files

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
PYTHONPATH := .

.PHONY: help
help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install in editable mode with dev extras.
	pip install -e ".[dev]"

# ============================================================================
# Tests
# ============================================================================

.PHONY: test
test:  ## Run pytest + reproducer + self-test + 10 examples.
	@echo "=== pytest ==="
	@PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q
	@echo ""
	@echo "=== run_audit.py (case-study reproducer) ==="
	@$(PYTHON) run_audit.py
	@echo ""
	@echo "=== self_test (synthetic ground truth) ==="
	@$(PYTHON) benchmarks/self_test.py
	@echo ""
	@echo "=== 10 examples ==="
	@for f in examples/recipe*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done

.PHONY: pytest
pytest:  ## Run pytest only (fast, ~0.2 s).
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q

.PHONY: reproducer
reproducer:  ## Run the case-study reproducer only.
	$(PYTHON) run_audit.py

.PHONY: self-test
self-test:  ## Run the synthetic-ground-truth self-test.
	$(PYTHON) benchmarks/self_test.py

.PHONY: examples
examples:  ## Run all 10 standalone example recipes.
	@for f in examples/recipe*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done

# ============================================================================
# Lint
# ============================================================================

.PHONY: lint
lint:  ## Run ruff check.
	ruff check eval_trust/ tests/ run_audit.py papers/eval_trust/scripts/ benchmarks/ examples/

.PHONY: lint-fix
lint-fix:  ## Run ruff with --fix.
	ruff check --fix eval_trust/ tests/ run_audit.py papers/eval_trust/scripts/ benchmarks/ examples/

# ============================================================================
# Paper artifacts
# ============================================================================

.PHONY: figures
figures:  ## Regenerate paper figures from data/.
	$(PYTHON) papers/eval_trust/scripts/make_figures.py

.PHONY: paper
paper: figures  ## Rebuild draft.pdf + arxiv_upload.tar.gz (requires pandoc + tectonic).
	bash papers/eval_trust/scripts/prepare_arxiv.sh

.PHONY: paper-fast
paper-fast: figures  ## Rebuild arxiv tar without compile sanity check (no tectonic needed).
	bash papers/eval_trust/scripts/prepare_arxiv.sh --no-compile

# ============================================================================
# Aggregates
# ============================================================================

.PHONY: all
all: test lint figures paper  ## test + lint + figures + paper.

.PHONY: ci
ci: pytest lint reproducer self-test examples  ## What CI runs (no paper compile).

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:  ## Remove generated files (build dirs, __pycache__, .pytest_cache).
	rm -rf papers/eval_trust/arxiv_upload/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

.PHONY: distclean
distclean: clean  ## Also remove built artifacts (figures, draft.pdf, arxiv tar).
	rm -f papers/eval_trust/figures/*.{pdf,png}
	rm -f papers/eval_trust/draft.pdf
	rm -f papers/eval_trust/arxiv_upload.tar.gz

# Default target
.DEFAULT_GOAL := help
