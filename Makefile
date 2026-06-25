# Makefile — common dev tasks for trace-pipeline
#
# Usage:
#   make help          # show all targets
#   make test          # pytest + reproducer + self-test + all examples
#   make lint          # ruff check (eval_trust + evomerge + scripts + tests)
#   make schema-check  # verify schemas/ and field coverage
#   make schemas       # regenerate schemas/*.schema.json
#   make figures       # regenerate paper figures from data/
#   make paper         # rebuild draft.pdf + arxiv_upload.tar.gz
#   make all           # test + lint + schema-check + figures + paper
#   make clean         # remove generated files

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
PYTHONPATH := .

.PHONY: help
help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install in editable mode with dev extras.
	pip install -e ".[dev]"

.PHONY: install-ml
install-ml:  ## Install ML extras (scikit-learn, joblib).
	pip install -e ".[dev,ml]"

.PHONY: install-train
install-train:  ## Install full training stack (peft, trl, transformers, bitsandbytes).
	pip install -e ".[dev,ml,train]"

# ============================================================================
# Tests
# ============================================================================

.PHONY: test
test:  ## Run pytest + reproducer + self-test + all examples (eval_trust + pipeline).
	@echo "=== pytest ==="
	@PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q
	@echo ""
	@echo "=== run_audit.py (case-study reproducer) ==="
	@$(PYTHON) run_audit.py
	@echo ""
	@echo "=== self_test (synthetic ground truth) ==="
	@$(PYTHON) benchmarks/self_test.py
	@echo ""
	@echo "=== eval_trust examples (recipe1-10) ==="
	@for f in examples/recipe{1..10}*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done
	@echo ""
	@echo "=== evomerge pipeline examples (recipe11-16) ==="
	@for f in examples/recipe1{1..6}*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done

.PHONY: pytest
pytest:  ## Run pytest only (fast, ~1 s).
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q

.PHONY: reproducer
reproducer:  ## Run the case-study reproducer only.
	$(PYTHON) run_audit.py

.PHONY: self-test
self-test:  ## Run the synthetic-ground-truth self-test.
	$(PYTHON) benchmarks/self_test.py

.PHONY: examples
examples:  ## Run all standalone example recipes (eval_trust + pipeline).
	@for f in examples/recipe*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done

# ============================================================================
# Lint
# ============================================================================

.PHONY: lint
lint:  ## Run ruff check (eval_trust, evomerge, scripts, tests, examples).
	ruff check eval_trust/ evomerge/ scripts/ tests/ run_audit.py \
		papers/eval_trust/scripts/ benchmarks/ examples/

.PHONY: lint-fix
lint-fix:  ## Run ruff with --fix.
	ruff check --fix eval_trust/ evomerge/ scripts/ tests/ run_audit.py \
		papers/eval_trust/scripts/ benchmarks/ examples/

# ============================================================================
# Schema tooling
# ============================================================================

.PHONY: schema-check
schema-check:  ## Verify schemas/ and data-loop field coverage.
	@echo "=== data-loop field coverage ==="
	@$(PYTHON) scripts/check-schema-fields.py
	@echo ""
	@echo "=== JSON Schema export check ==="
	@$(PYTHON) scripts/export-schemas.py --check

.PHONY: schemas
schemas:  ## Regenerate schemas/*.schema.json from Pydantic models.
	$(PYTHON) scripts/export-schemas.py

.PHONY: sync-schemas
sync-schemas:  ## Sync schemas from wasmagent-js (pass WASMAGENT_JS=/path/to/repo).
	@if [ -z "$(WASMAGENT_JS)" ]; then \
		echo "Usage: make sync-schemas WASMAGENT_JS=/path/to/wasmagent-js"; \
		exit 1; \
	fi
	$(PYTHON) scripts/sync-wasmagent-schemas.py --wasmagent-js $(WASMAGENT_JS)

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
all: test lint schema-check figures paper  ## test + lint + schema-check + figures + paper.

.PHONY: ci
ci: pytest lint schema-check reproducer self-test examples  ## What CI runs (no paper compile).

# ============================================================================
# Pipeline CLI shortcuts
# ============================================================================

.PHONY: export
export:  ## Export training JSONL from fixture (demo). Pass ROLLOUT= to use real data.
	$(PYTHON) -m evomerge export \
		--rollout $${ROLLOUT:-fixtures/data-loop/rollout-branches.v1.jsonl} \
		--out-dir $${OUT_DIR:-/tmp/evomerge-export}
	@echo "manifest: $${OUT_DIR:-/tmp/evomerge-export}/manifest.json"

.PHONY: validate-export
validate-export:  ## Validate the last export output.
	$(PYTHON) -m evomerge validate \
		--input $${OUT_DIR:-/tmp/evomerge-export}/sft.jsonl --strict

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
