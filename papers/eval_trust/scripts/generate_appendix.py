"""papers/eval_trust/scripts/generate_appendix.py — Generate paper appendix (Markdown).

Sections:
  A. Reproducibility (commands + SHA-256 hashes of key data files)
  B. Key Results table
  C. Environment

Key results are hardcoded from CLAUDE.md benchmark summary:
  - IFEval x Qwen2.5-1.5B-Q4: full_pcl 54.7 +/- 1.2%, prompt_retry 46.0 +/- 2.0%, +8.7 pp

Usage:
  python papers/eval_trust/scripts/generate_appendix.py              # stdout
  python papers/eval_trust/scripts/generate_appendix.py --output papers/eval_trust/APPENDIX.md
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "papers" / "eval_trust"
DATA_DIR = ROOT / "data"

# Data files to include in the reproducibility table (relative to ROOT)
TRACKED_DATA_FILES = [
    "data/audit_length_analysis.json",
    "data/t0v2_aggregate.json",
    "data/self_consistency_full.json",
    "data/instruct_max_new768.json",
    "data/winner_max_new768.json",
    "data/gsm8k_dev_200.json",
    "data/quantization_granularity/summary.json",
    "data/synthetic_4algo/marginal_history_anonymized.json",
]

# Hardcoded key results from CLAUDE.md
KEY_RESULTS = [
    {
        "model": "Qwen2.5-1.5B-Q4",
        "benchmark": "IFEval",
        "method": "full_pcl",
        "pass_rate": "54.7%",
        "std": "±1.2%",
        "seeds": 3,
        "note": "3 seeds × 50 samples",
    },
    {
        "model": "Qwen2.5-1.5B-Q4",
        "benchmark": "IFEval",
        "method": "prompt_retry",
        "pass_rate": "46.0%",
        "std": "±2.0%",
        "seeds": 3,
        "note": "baseline",
    },
    {
        "model": "Llama-3.2-1B",
        "benchmark": "IFEval",
        "method": "full_pcl",
        "pass_rate": "ties prompt_retry",
        "std": "5× smaller variance",
        "seeds": 3,
        "note": "PCL reduces variance",
    },
]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, or 'FILE NOT FOUND' if missing."""
    if not path.exists():
        return "FILE NOT FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_str(path: Path) -> str:
    """Return human-readable file size, or '-' if missing."""
    if not path.exists():
        return "-"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_appendix() -> str:
    lines: list[str] = []

    def h(text: str) -> None:
        lines.append(text)

    def blank() -> None:
        lines.append("")

    # ── Header ──────────────────────────────────────────────────────────────
    h("# Appendix")
    blank()
    h("This appendix provides reproducibility information, key numeric results, "
      "and environment details for the eval-trust paper.")
    blank()

    # ── A. Reproducibility ───────────────────────────────────────────────────
    h("## A  Reproducibility")
    blank()
    h("### A.1  Regenerating figures")
    blank()
    h("All figures are generated from data files committed in this repository. "
      "No external project state is required.")
    blank()
    h("```bash")
    h("# From repo root")
    h("python papers/eval_trust/scripts/make_figures.py")
    h("# Outputs: papers/eval_trust/figures/{fig1,fig2,fig3}.{pdf,png}")
    h("```")
    blank()
    h("### A.2  Regenerating this appendix")
    blank()
    h("```bash")
    h("python papers/eval_trust/scripts/generate_appendix.py \\")
    h("    --output papers/eval_trust/APPENDIX.md")
    h("```")
    blank()
    h("### A.3  Compliance benchmark sweep (IFEval)")
    blank()
    h("Reproduces the IFEval × Qwen2.5-1.5B-Q4 results in Table B.1 below.")
    blank()
    h("```bash")
    h("# From wasmagent-js repo root — runs 3 seeds × 50 samples")
    h("bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=42")
    h("bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=43")
    h("bun packages/compliance/benchmarks/ifeval/run.ts --limit=50 --seed=44")
    blank()
    h("# Aggregate across seeds")
    h("bun packages/compliance/benchmarks/ifeval/compare-seeds.ts")
    h("```")
    blank()
    h("### A.4  Data file checksums")
    blank()
    h("SHA-256 digests of all tracked data files at time of appendix generation.")
    blank()
    h("| File | Size | SHA-256 |")
    h("|------|------|---------|")
    for rel in TRACKED_DATA_FILES:
        path = ROOT / rel
        digest = sha256_file(path)
        size = file_size_str(path)
        short = digest[:16] + "..." if digest != "FILE NOT FOUND" else digest
        h(f"| `{rel}` | {size} | `{short}` |")
    blank()

    # ── B. Key Results ───────────────────────────────────────────────────────
    h("## B  Key Results")
    blank()
    h("### B.1  IFEval pass-rate summary")
    blank()
    h("Results from the Compliance Engine PCL evaluation "
      "(3 seeds × 50 samples per condition).")
    blank()
    h("| Model | Benchmark | Method | Pass-rate | Std | Notes |")
    h("|-------|-----------|--------|-----------|-----|-------|")
    for r in KEY_RESULTS:
        h(f"| {r['model']} | {r['benchmark']} | `{r['method']}` "
          f"| {r['pass_rate']} | {r['std']} | {r['note']} |")
    blank()
    h("**Headline finding**: on IFEval × Qwen2.5-1.5B-Q4, `full_pcl` achieves "
      "**54.7% ± 1.2%** vs `prompt_retry` **46.0% ± 2.0%** "
      "(**+8.7 pp**, 3 seeds × 50 samples).")
    blank()
    h("On Llama-3.2-1B, PCL ties `prompt_retry` on mean but has 5× smaller variance, "
      "suggesting PCL primarily reduces stochastic failures on harder models.")
    blank()
    h("### B.2  Result data location")
    blank()
    h("| Description | Path (wasmagent-js repo) |")
    h("|-------------|-------------------------|")
    h("| Raw records (1 050 entries) | `packages/compliance/benchmarks/ifeval/results*/` |")
    h("| Multi-seed phase reports | `packages/compliance/benchmarks/ifeval/results-multi-seed*/*.md` |")
    h("| Cross-model summary | `packages/compliance/benchmarks/ifeval/results-multi-seed-llama/CROSS-MODEL-2026-06-24.md` |")
    blank()

    # ── C. Environment ───────────────────────────────────────────────────────
    h("## C  Environment")
    blank()
    h("### C.1  Runtime")
    blank()
    h("| Component | Version / value |")
    h("|-----------|-----------------|")
    h("| Runtime | Bun ≥ 1.2 |")
    h("| Language | TypeScript 5 |")
    h("| Package manager | npm workspaces + turbo |")
    h("| Test runner | `bun test` |")
    h("| Linter | Biome |")
    blank()
    h("### C.2  Model environment")
    blank()
    h("| Parameter | Value |")
    h("|-----------|-------|")
    h("| Qwen2.5-1.5B quantization | Q4 (GGUF, per-tensor) |")
    h("| Llama-3.2-1B quantization | Q4 (GGUF, per-tensor) |")
    h("| IFEval instruction classes | 15 |")
    h("| Compliance verifier built-in checks | 7 |")
    h("| Repair strategies | PatchStrategy, InsertSectionStrategy, RegenerateRegionStrategy |")
    blank()
    h("### C.3  Relevant packages")
    blank()
    h("| Package | Stability |")
    h("|---------|-----------|")
    h("| `@wasmagent/compliance` | Alpha (schema versioned) |")
    h("| `@wasmagent/core` | Stable |")
    h("| `@wasmagent/kernel-quickjs` | Stable |")
    h("| `@wasmagent/evals-runner` | Growth |")
    blank()

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper appendix (Markdown).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path (default: stdout, use - for stdout)",
    )
    args = parser.parse_args()

    content = generate_appendix()

    if args.output in ("-", ""):
        sys.stdout.write(content + "\n")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content + "\n", encoding="utf-8")
        print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
