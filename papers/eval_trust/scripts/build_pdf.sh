#!/bin/bash
# build_pdf.sh — 一键 md → LaTeX → PDF
# 依赖: pandoc + tectonic (前者负责 md→tex 转换, 后者编 tex→pdf 自动下载 latex pkg)
#
# 使用:
#   bash papers/eval_trust/scripts/build_pdf.sh
#
# 输出:
#   papers/eval_trust/draft.tex (中间 LaTeX 源, 可直接喂给 arxiv)
#   papers/eval_trust/draft.pdf (最终 PDF, ~230 KB)
set -e
cd "$(dirname "$0")/../../.."  # back to repo root

PROXY="${PROXY:-http://proxy.sin.sap.corp:8080}"

echo "[1/2] pandoc md → tex"
pandoc papers/eval_trust/draft.md \
    --from markdown+raw_tex+citations+grid_tables+pipe_tables \
    --to latex \
    --bibliography papers/eval_trust/refs.bib \
    --citeproc \
    --standalone \
    --metadata title="Silent Contamination in LLM Merging Evaluation: A Case Study from a 5-Month Misadventure" \
    --metadata author="telleroutlook (evomerge project)" \
    --metadata date="2026-06-05" \
    -o papers/eval_trust/draft.tex

echo "[2/2] tectonic tex → pdf (auto-downloads latex packages on first run)"
cd papers/eval_trust
https_proxy="$PROXY" http_proxy="$PROXY" \
    tectonic -X compile draft.tex 2>&1 | tail -3

ls -lah draft.pdf
echo "Done. PDF: papers/eval_trust/draft.pdf"
