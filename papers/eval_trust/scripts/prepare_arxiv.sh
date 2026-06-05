#!/bin/bash
# papers/eval_trust/scripts/prepare_arxiv.sh
#
# Build an arxiv-ready upload tar.gz from papers/eval_trust source.
#
# What it does:
#   1. pandoc papers/eval_trust/draft.md -> arxiv build dir as main.tex
#      (with --citeproc so all [@key] resolve inline; no bibtex re-run on arxiv)
#   2. Copies refs.bib + 3 figure PDFs into the build dir
#   3. Flattens figure paths in main.tex (figures/X.pdf -> X.pdf)
#   4. Optionally compiles main.pdf via tectonic as a sanity check
#   5. Tars build dir into papers/eval_trust/arxiv_upload.tar.gz
#
# Idempotent: safe to rerun. Re-overwrites the build dir + tar.
#
# Usage from repo root:
#   bash papers/eval_trust/scripts/prepare_arxiv.sh
#   bash papers/eval_trust/scripts/prepare_arxiv.sh --no-compile  # skip tectonic
#
# Requirements:
#   - pandoc (for md -> tex)
#   - tectonic (for sanity-compile; optional with --no-compile)
#
# Exit codes:
#   0 = success (tar.gz produced + PDF compile passed if --no-compile not set)
#   1 = pandoc/tectonic missing or any step failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

BUILD_DIR="$PAPER_DIR/arxiv_upload"
OUT_TAR="$PAPER_DIR/arxiv_upload.tar.gz"

COMPILE=true
for arg in "$@"; do
    if [ "$arg" = "--no-compile" ]; then
        COMPILE=false
    fi
done

# Pre-flight
if ! command -v pandoc >/dev/null 2>&1; then
    echo "[ERR] pandoc not found. Install: brew install pandoc"
    exit 1
fi
if [ "$COMPILE" = "true" ] && ! command -v tectonic >/dev/null 2>&1; then
    echo "[WARN] tectonic not found; skipping compile sanity check."
    echo "       Install: brew install tectonic, or rerun with --no-compile."
    COMPILE=false
fi

echo "[1/5] cleaning build dir..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[2/5] pandoc md -> tex..."
pandoc "$PAPER_DIR/draft.md" \
    --from markdown+raw_tex+citations+grid_tables+pipe_tables \
    --to latex \
    --bibliography "$PAPER_DIR/refs.bib" \
    --citeproc \
    --standalone \
    --metadata title="Silent Contamination in LLM Merging Evaluation: A Case Study from a 5-Month Misadventure" \
    --metadata author="telleroutlook (evomerge project)" \
    --metadata date="$(date +%Y-%m-%d)" \
    -o "$BUILD_DIR/main.tex"

echo "[3/5] copying refs + figures..."
cp "$PAPER_DIR/refs.bib" "$BUILD_DIR/refs.bib"
for fig in "$PAPER_DIR/figures"/fig*.pdf; do
    cp "$fig" "$BUILD_DIR/$(basename "$fig")"
done

echo "[4/5] flattening figure paths in main.tex..."
# arxiv uses a flat upload — strip the figures/ prefix.
python3 -c "
import re
from pathlib import Path
p = Path('$BUILD_DIR/main.tex')
text = p.read_text()
# Match figures/<anything>.pdf — broad pattern, OK because tex paths are
# always inside a \includegraphics{...} so collateral damage is unlikely.
text = re.sub(r'figures/([^/{}\s]+\.pdf)', r'\1', text)
p.write_text(text)
print('  patched paths in main.tex')
"

if [ "$COMPILE" = "true" ]; then
    echo "[5a/5] tectonic sanity-compile..."
    pushd "$BUILD_DIR" >/dev/null
    if tectonic -X compile main.tex 2>&1 | tail -3; then
        echo "  PDF size: $(ls -lah main.pdf | awk '{print $5}')"
    else
        echo "[ERR] tectonic compile failed"
        popd >/dev/null
        exit 1
    fi
    popd >/dev/null

    # Refresh the public-facing draft.pdf from the just-compiled PDF.
    # This keeps the repo's draft.pdf in lockstep with draft.md any time
    # this script runs.
    cp "$BUILD_DIR/main.pdf" "$PAPER_DIR/draft.pdf"
    echo "  refreshed $PAPER_DIR/draft.pdf"
fi

echo "[5b/5] tarring..."
rm -f "$OUT_TAR"
# Exclude the sanity-check PDF; arxiv compiles its own.
# tar from inside build dir so paths are flat.
tar -czf "$OUT_TAR" -C "$BUILD_DIR" --exclude=main.pdf .

ls -lah "$OUT_TAR"
echo ""
echo "  ===== contents ====="
tar -tzf "$OUT_TAR" | sort

echo ""
echo "  Done. Upload $OUT_TAR to arxiv.org submission form."
