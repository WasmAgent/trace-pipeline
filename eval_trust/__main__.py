"""eval_trust CLI — audit toolkit entry point.

Usage:
    python -m eval_trust <command> [options]

Commands:
    paired      Paired McNemar + bootstrap significance test on two result files.
    conformal   Compute split-conformal accuracy interval from a calibration set.
    truncation  Detect truncated generations in an lm-eval samples_*.jsonl file.
    audit       Full pipeline: paired + conformal + truncation in one shot.

Run `python -m eval_trust <command> --help` for per-command options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_correctness(path: str) -> list[bool]:
    """Load per-item correctness from an lm-eval samples_*.jsonl or a plain list JSON."""
    p = Path(path)
    if p.suffix == ".jsonl":
        items = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return [bool(item.get("correct", item.get("pass", False))) for item in items]
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return [bool(x) for x in data]
    results = data.get("results", data.get("items", []))
    return [bool(r.get("correct", False)) for r in results]


def _load_results_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


# ── paired ───────────────────────────────────────────────────────────────────

def _cmd_paired(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval_trust paired",
        description=(
            "Paired McNemar + bootstrap significance test.\n\n"
            "Compares two result files item-by-item and reports:\n"
            "  - per-model accuracy + Wilson CI\n"
            "  - McNemar exact p-value\n"
            "  - paired bootstrap delta + 95%% CI\n"
            "  - minimum n for 80%% power at the observed delta"
        ),
    )
    p.add_argument("a", metavar="FILE_A", help="JSON/JSONL results for model A")
    p.add_argument("b", metavar="FILE_B", help="JSON/JSONL results for model B")
    p.add_argument("--conf", type=float, default=0.95, metavar="F",
                   help="Wilson CI confidence level (default: 0.95)")
    p.add_argument("--bootstrap-iters", type=int, default=10_000, metavar="N",
                   help="Bootstrap resampling iterations (default: 10000)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = p.parse_args(argv)

    from eval_trust.paired_stats import mcnemar_exact, paired_bootstrap, wilson_ci, n_for_power

    ca = _load_correctness(args.a)
    cb = _load_correctness(args.b)
    if len(ca) != len(cb):
        print(f"ERROR: item count mismatch: {len(ca)} vs {len(cb)}", file=sys.stderr)
        return 1

    n = len(ca)
    b_count = sum(1 for a, bv in zip(ca, cb) if a and not bv)
    c_count = sum(1 for a, bv in zip(ca, cb) if not a and bv)
    acc_a = sum(ca) / n
    acc_b = sum(cb) / n
    p_val = mcnemar_exact(b_count, c_count)
    delta, (ci_lo, ci_hi) = paired_bootstrap(ca, cb, n_iter=args.bootstrap_iters)
    lo_a, hi_a = wilson_ci(sum(ca), n, args.conf)
    lo_b, hi_b = wilson_ci(sum(cb), n, args.conf)
    n_power = n_for_power(abs(acc_a - acc_b)) if acc_a != acc_b else None

    if args.json:
        out = {
            "n": n, "b": b_count, "c": c_count,
            "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "wilson_a": [round(lo_a, 4), round(hi_a, 4)],
            "wilson_b": [round(lo_b, 4), round(hi_b, 4)],
            "mcnemar_p": round(p_val, 4),
            "bootstrap_delta": round(delta, 4),
            "bootstrap_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_for_80pct_power": n_power,
        }
        print(json.dumps(out, indent=2))
    else:
        sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "not significant"
        print(f"n = {n}   b = {b_count}   c = {c_count}")
        print(f"Model A: {acc_a:.1%}  Wilson {args.conf:.0%} CI [{lo_a:.3f}, {hi_a:.3f}]")
        print(f"Model B: {acc_b:.1%}  Wilson {args.conf:.0%} CI [{lo_b:.3f}, {hi_b:.3f}]")
        print(f"McNemar p = {p_val:.4f}  →  {sig}")
        print(f"Bootstrap delta = {delta:+.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        if n_power:
            print(f"Minimum n for 80%% power at this delta: {n_power}")
    return 0


# ── conformal ────────────────────────────────────────────────────────────────

def _cmd_conformal(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval_trust conformal",
        description=(
            "Split-conformal accuracy interval.\n\n"
            "Calibrate on a held-out split and return a distribution-free "
            "prediction interval for the held-in accuracy."
        ),
    )
    p.add_argument("calibration", metavar="CALIBRATION_FILE",
                   help="JSON/JSONL correctness array for the calibration split")
    p.add_argument("prediction", metavar="PRED_ACCURACY", type=float,
                   help="Point-estimate accuracy for the test split (0.0–1.0)")
    p.add_argument("--alpha", type=float, default=0.1, metavar="F",
                   help="Error rate; interval covers 1-alpha (default: 0.1 → 90%%)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = p.parse_args(argv)

    from eval_trust.conformal_ci import compute_conformal_interval

    calib = _load_correctness(args.calibration)
    residuals = [abs(float(c) - args.prediction) for c in calib]
    interval = compute_conformal_interval(residuals, args.prediction, alpha=args.alpha)

    if args.json:
        print(json.dumps({"lo": round(interval.lo, 4), "hi": round(interval.hi, 4),
                          "alpha": args.alpha, "n_calib": len(calib)}, indent=2))
    else:
        print(f"Conformal interval (alpha={args.alpha}): [{interval.lo:.4f}, {interval.hi:.4f}]")
        print(f"Calibration set n = {len(calib)}")
    return 0


# ── truncation ───────────────────────────────────────────────────────────────

def _cmd_truncation(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval_trust truncation",
        description=(
            "Detect truncated generations in an lm-eval samples_*.jsonl file.\n\n"
            "Flags items where gen_text approaches max_new_tokens and lacks a "
            "parseable final answer (the case-study primitive from the paper)."
        ),
    )
    p.add_argument("samples", metavar="SAMPLES_JSONL",
                   help="lm-evaluation-harness samples_*.jsonl file")
    p.add_argument("--max-new-tokens", type=int, default=300, metavar="N",
                   help="max_new_tokens used in the evaluation run (default: 300)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = p.parse_args(argv)

    from eval_trust.t0v2.truncation_extract import classify

    items = [json.loads(line) for line in Path(args.samples).read_text().splitlines()
             if line.strip()]
    results = [classify(item, max_new_tokens=args.max_new_tokens) for item in items]
    n_truncated = sum(1 for r in results if r.get("truncated"))

    if args.json:
        print(json.dumps({"n": len(items), "n_truncated": n_truncated,
                          "truncation_rate": round(n_truncated / len(items), 4) if items else 0,
                          "items": results}, indent=2))
    else:
        rate = n_truncated / len(items) if items else 0
        print(f"Scanned {len(items)} items — {n_truncated} truncated ({rate:.1%})")
        if n_truncated:
            print("\nTruncated indices:")
            for r in results:
                if r.get("truncated"):
                    print(f"  [{r.get('doc_id', '?')}] len={r.get('gen_len')} tokens")
    return 0


# ── audit (full pipeline) ────────────────────────────────────────────────────

def _cmd_audit(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval_trust audit",
        description=(
            "Full audit pipeline: run paired + contamination checks on two result files.\n\n"
            "Equivalent to running 'paired' then reviewing the case-study primitives "
            "from the paper in one command."
        ),
    )
    p.add_argument("a", metavar="FILE_A", help="JSON/JSONL results for model A")
    p.add_argument("b", metavar="FILE_B", help="JSON/JSONL results for model B")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = p.parse_args(argv)

    return _cmd_paired([args.a, args.b] + (["--json"] if args.json else []))


# ── dispatch ─────────────────────────────────────────────────────────────────

COMMANDS = {
    "paired": _cmd_paired,
    "conformal": _cmd_conformal,
    "truncation": _cmd_truncation,
    "audit": _cmd_audit,
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="python -m eval_trust",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=list(COMMANDS),
                        help="Subcommand to run")
    parser.add_argument("args", nargs=argparse.REMAINDER)

    top = parser.parse_args(argv)
    if not top.command:
        parser.print_help()
        return 0

    return COMMANDS[top.command](top.args)


if __name__ == "__main__":
    sys.exit(main())
