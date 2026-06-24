#!/usr/bin/env python3
"""Train and evaluate a Router ML classifier on real IFEval benchmark data.

Strategy:
  Features  — extracted from the 'direct' mode attempt of each task
              (what the router sees after a single small-model call)
  Labels    — derived by comparing direct vs full_pcl outcomes:
                direct passes              → small_model_can_handle
                direct fails, pcl passes   → need_repair
                direct fails, pcl fails    → need_large_model

  Classifier — GradientBoostingClassifier (sklearn), 5-fold stratified CV
  Output     — printed report + saved model (joblib) + saved RouterRecord JSONL

Usage:
    python scripts/train_router.py \\
        --runs-dir /path/to/wasmagent-js/packages/compliance/benchmarks/ifeval \\
        --out-dir  data/router

    # skip saving the model (report only)
    python scripts/train_router.py --runs-dir ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

BENCHMARK_DIRS = [
    ("results",                     "qwen2.5-1.5b"),
    ("results-seed43",              "qwen2.5-1.5b"),
    ("results-seed44",              "qwen2.5-1.5b"),
    ("results-llama-3.2-1b-seed42", "llama-3.2-1b"),
    ("results-llama-3.2-1b-seed43", "llama-3.2-1b"),
    ("results-llama-3.2-1b-seed44", "llama-3.2-1b"),
]

FEATURE_NAMES = [
    "taskspec_has_tools",    # 1 if task requires tool calls (from TaskSpec; 0 for IFEval)
    "taskspec_n_allowed_tools",  # number of allowed tools (0 for IFEval)
    "n_violations",
    "n_hard_violations",
    "n_soft_violations",
    "n_format_violations",
    "n_content_violations",
    "repair_rounds",
    "n_repair_ok",
    "n_repair_fail",
    "latency_ms",
    "prompt_tokens",
    "generation_tokens",
    "token_ratio",           # generation / (prompt + 1)
    "model_is_qwen",         # 1 for qwen, 0 for llama
]


# ── feature extraction ────────────────────────────────────────────────────────

@dataclass
class Sample:
    task_id: str
    model: str
    subdir: str
    features: list[float]
    label: str
    direct_pass: bool
    pcl_pass: bool


def _extract_features(direct_rec: dict) -> list[float]:
    violations = direct_rec.get("violations", [])
    repair_trace = direct_rec.get("repair_trace", [])
    tok = direct_rec.get("token_cost", {}) or {}

    n_v        = len(violations)
    n_hard     = sum(1 for v in violations if v.get("level") == "hard")
    n_soft     = n_v - n_hard
    n_format   = sum(1 for v in violations if v.get("category") == "format")
    n_content  = sum(1 for v in violations if v.get("category") == "content")
    repair_rounds = direct_rec.get("repair_rounds", 0)
    n_ok       = sum(1 for e in repair_trace if e.get("ok"))
    n_fail     = len(repair_trace) - n_ok
    latency    = float(direct_rec.get("latency_ms", 0))
    prompt_tok = float(tok.get("prompt") or 0)
    gen_tok    = float(tok.get("generation") or 0)
    tok_ratio  = gen_tok / (prompt_tok + 1)
    is_qwen    = 1.0 if "qwen" in direct_rec.get("model", "").lower() else 0.0

    return [
        0.0, 0.0,              # taskspec_has_tools, taskspec_n_allowed_tools (IFEval has no tools)
        n_v, n_hard, n_soft, n_format, n_content,
        repair_rounds, n_ok, n_fail,
        latency, prompt_tok, gen_tok, tok_ratio,
        is_qwen,
    ]


def _derive_label(direct_rec: dict, pcl_rec: dict) -> str:
    direct_pass = direct_rec.get("final_pass", False)
    pcl_pass    = pcl_rec.get("final_pass", False)
    pcl_repair  = pcl_rec.get("repair_rounds", 0)

    if direct_pass:
        return "small_model_can_handle"
    if pcl_pass:
        return "need_repair"
    return "need_large_model"


def load_samples(runs_dir: Path) -> list[Sample]:
    # key: (task_id, model, subdir) → {mode: record}
    by_key: dict[tuple, dict] = defaultdict(dict)
    for subdir, model in BENCHMARK_DIRS:
        jsonl = runs_dir / subdir / "runs.jsonl"
        if not jsonl.exists():
            print(f"[skip] {jsonl}", file=sys.stderr)
            continue
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["task_id"], r["model"], subdir)
                by_key[key][r["mode"]] = r

    samples: list[Sample] = []
    for (task_id, model, subdir), modes in by_key.items():
        direct = modes.get("direct")
        pcl    = modes.get("full_pcl")
        if direct is None or pcl is None:
            continue
        samples.append(Sample(
            task_id=task_id,
            model=model,
            subdir=subdir,
            features=_extract_features(direct),
            label=_derive_label(direct, pcl),
            direct_pass=direct.get("final_pass", False),
            pcl_pass=pcl.get("final_pass", False),
        ))
    return samples


# ── training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(samples: list[Sample], dry_run: bool = False) -> dict:
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
        from collections import Counter
    except ImportError as exc:
        print(f"[error] {exc}\n  pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    X = np.array([s.features for s in samples], dtype=np.float32)
    y_raw = [s.label for s in samples]

    label_order = ["small_model_can_handle", "need_repair", "need_large_model"]
    le = LabelEncoder()
    le.fit(label_order)
    y = le.transform(y_raw)

    print(f"\n  samples  : {len(samples)}")
    print(f"  features : {len(FEATURE_NAMES)}")
    label_dist = Counter(y_raw)
    for lbl in label_order:
        n = label_dist[lbl]
        print(f"  {lbl:<30} {n:>4}  ({n/len(samples):.1%})")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True,
    )

    print("\n  5-fold cross-validation:")
    for metric in ["test_accuracy", "test_f1_macro", "test_f1_weighted"]:
        scores = cv_results[metric]
        print(f"    {metric:<22} {scores.mean():.4f} ± {scores.std():.4f}  "
              f"[{scores.min():.4f}–{scores.max():.4f}]")

    # train on full dataset for final model + feature importance
    clf.fit(X, y)
    y_pred = clf.predict(X)

    print("\n  training set classification report:")
    print(classification_report(y, y_pred, target_names=le.classes_, digits=3))

    print("  confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred)
    header = "  " + "  ".join(f"{c[:8]:>10}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {le.classes_[i][:8]:>10}  {row_str}")

    print("\n  feature importances (top 10):")
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx[:10], 1):
        print(f"    {rank:>2}. {FEATURE_NAMES[i]:<25} {importances[i]:.4f}")

    result = {
        "n_samples": len(samples),
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std":  float(cv_results["test_accuracy"].std()),
        "cv_f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
        "cv_f1_macro_std":  float(cv_results["test_f1_macro"].std()),
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
        "feature_importances": {
            FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))
        },
    }

    return result, clf, le


# ── save RouterRecord JSONL ───────────────────────────────────────────────────

def save_router_records(samples: list[Sample], out_path: Path) -> None:
    from evomerge.router.labels import RouterLabel, RouterRecord
    from evomerge.router.features import RouterFeatures
    from evomerge.schemas.training import Provenance
    import hashlib

    label_map = {
        "small_model_can_handle": RouterLabel.small_model_can_handle,
        "need_repair":            RouterLabel.need_repair,
        "need_large_model":       RouterLabel.need_large_model,
    }

    records = []
    for s in samples:
        f = s.features
        # FEATURE_NAMES order: has_tools(0), n_allowed_tools(1), n_v(2), n_hard(3),
        # n_soft(4), n_format(5), n_content(6), repair_rounds(7), n_ok(8), n_fail(9),
        # latency(10), prompt_tok(11), gen_tok(12), tok_ratio(13), is_qwen(14)
        features = RouterFeatures(
            taskspec_n_constraints=int(f[2]),        # n_violations as proxy
            taskspec_n_hard=int(f[3]),
            taskspec_has_tools=int(f[0]),
            taskspec_n_allowed_tools=int(f[1]),
            taskspec_max_repair_rounds=3,
            eval_repair_rounds=int(f[7]),
            eval_violation_count=int(f[2]),
            eval_hard_violation_count=int(f[3]),
            eval_tool_calls_total=0,
            eval_tool_calls_valid=0,
            eval_tool_validity_rate=1.0,
            eval_escalated=0,
            eval_latency_ms=float(f[10]),
            eval_prompt_tokens=int(f[11]),
            eval_generation_tokens=int(f[12]),
        )
        records.append(RouterRecord(
            task_id=f"{s.task_id}@{s.subdir}",
            features=features,
            label=label_map[s.label],
            provenance=Provenance(
                source="wasmagent-ifeval",
                task_id=s.task_id,
                task_hash=hashlib.sha256(s.task_id.encode()).hexdigest()[:16],
            ),
        ))

    from evomerge.io import write_dicts_jsonl
    write_dicts_jsonl([r.to_dict() for r in records], out_path)
    print(f"  wrote {len(records)} RouterRecord → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", required=True, metavar="DIR")
    ap.add_argument("--out-dir",  default="data/router", metavar="DIR")
    ap.add_argument("--dry-run",  action="store_true",
                    help="print report only, do not save model or records")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[error] not found: {runs_dir}", file=sys.stderr)
        return 1

    print("loading samples...")
    samples = load_samples(runs_dir)
    if not samples:
        print("[error] no samples loaded", file=sys.stderr)
        return 1
    print(f"  loaded {len(samples)} (task, model, seed) triples")

    result, clf, le = train_and_evaluate(samples, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] model and records not saved")
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # save sklearn model
    try:
        import joblib
        model_path = out / "router_gbdt.joblib"
        joblib.dump({"clf": clf, "le": le, "feature_names": FEATURE_NAMES}, model_path)
        print(f"\n  saved model → {model_path}")
        result["model_path"] = str(model_path)
    except ImportError:
        print("\n[warn] joblib not available — model not saved (pip install joblib)")

    # save RouterRecord JSONL for downstream use
    rr_path = out / "router_records.jsonl"
    save_router_records(samples, rr_path)
    result["router_records_path"] = str(rr_path)

    # save report
    report_path = out / "training_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  saved report → {report_path}")

    print(f"\n✓ Router GBDT trained")
    print(f"  CV accuracy : {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
    print(f"  CV F1 macro : {result['cv_f1_macro_mean']:.4f} ± {result['cv_f1_macro_std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
