#!/usr/bin/env python3
"""Train a pre-run router baseline using only static TaskSpec features.

Unlike the post-attempt router (train_router.py) which observes violation counts
from the completed direct-mode run, this router uses ONLY features available
BEFORE any model call — i.e. features derivable from the TaskSpec alone.

Static TaskSpec features (5):
    taskspec_n_constraints    — total number of constraints in the TaskSpec
    taskspec_n_hard           — number of hard constraints
    taskspec_has_tools        — 1 if task allows any tools, else 0
    taskspec_n_allowed_tools  — number of allowed tools
    taskspec_max_repair_rounds — max repair rounds configured in the TaskSpec

Dynamic features (NOT used here, used in post-attempt router):
    eval_repair_rounds, eval_violation_count, eval_hard_violation_count,
    eval_tool_calls_total, eval_tool_calls_valid, eval_tool_validity_rate,
    eval_escalated, eval_latency_ms, eval_prompt_tokens, eval_generation_tokens

Usage:
    python scripts/train_router_prerun.py \\
        --records data/router/router_records.jsonl \\
        --out-dir  data/router
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

STATIC_FEATURE_NAMES = [
    "taskspec_n_constraints",
    "taskspec_n_hard",
    "taskspec_has_tools",
    "taskspec_n_allowed_tools",
    "taskspec_max_repair_rounds",
]


def load_records(records_path: Path) -> list[dict]:
    records = []
    with open(records_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_static_features(record: dict) -> list[float]:
    f = record["features"]
    return [
        float(f["taskspec_n_constraints"]),
        float(f["taskspec_n_hard"]),
        float(f["taskspec_has_tools"]),
        float(f["taskspec_n_allowed_tools"]),
        float(f["taskspec_max_repair_rounds"]),
    ]


def train_and_evaluate(records: list[dict]) -> dict:
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        print(f"[error] {exc}\n  pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    X = np.array([extract_static_features(r) for r in records], dtype=np.float32)
    y_raw = [r["label"] for r in records]

    label_order = ["small_model_can_handle", "need_repair", "need_large_model"]
    le = LabelEncoder()
    le.fit(label_order)
    y = le.transform(y_raw)

    print(f"\n  samples  : {len(records)}")
    print(f"  features : {len(STATIC_FEATURE_NAMES)}  (static TaskSpec only)")
    label_dist = Counter(y_raw)
    for lbl in label_order:
        n = label_dist[lbl]
        print(f"  {lbl:<30} {n:>4}  ({n/len(records):.1%})")

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

    print("\n  feature importances:")
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx, 1):
        print(f"    {rank:>2}. {STATIC_FEATURE_NAMES[i]:<35} {importances[i]:.4f}")

    result = {
        "n_samples": len(records),
        "n_features": len(STATIC_FEATURE_NAMES),
        "feature_type": "static_taskspec_only",
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std":  float(cv_results["test_accuracy"].std()),
        "cv_f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
        "cv_f1_macro_std":  float(cv_results["test_f1_macro"].std()),
        "cv_f1_weighted_mean": float(cv_results["test_f1_weighted"].mean()),
        "cv_f1_weighted_std":  float(cv_results["test_f1_weighted"].std()),
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
        "feature_importances": {
            STATIC_FEATURE_NAMES[i]: float(importances[i])
            for i in range(len(STATIC_FEATURE_NAMES))
        },
    }

    return result, clf, le


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--records",
        default="data/router/router_records.jsonl",
        metavar="JSONL",
        help="RouterRecord JSONL (default: data/router/router_records.jsonl)",
    )
    ap.add_argument(
        "--out-dir",
        default="data/router",
        metavar="DIR",
        help="Output directory for model and report (default: data/router)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report only, do not save model",
    )
    args = ap.parse_args()

    records_path = Path(args.records)
    if not records_path.is_absolute():
        records_path = REPO_ROOT / records_path
    if not records_path.exists():
        print(f"[error] not found: {records_path}", file=sys.stderr)
        return 1

    print(f"loading records from {records_path} ...")
    records = load_records(records_path)
    if not records:
        print("[error] no records loaded", file=sys.stderr)
        return 1
    print(f"  loaded {len(records)} RouterRecord entries")

    result, clf, le = train_and_evaluate(records)

    if args.dry_run:
        print("\n[dry-run] model not saved")
        return 0

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        model_path = out / "router_prerun_gbdt.joblib"
        joblib.dump(
            {"clf": clf, "le": le, "feature_names": STATIC_FEATURE_NAMES},
            model_path,
        )
        print(f"\n  saved model → {model_path}")
        result["model_path"] = str(model_path)
    except ImportError:
        print("\n[warn] joblib not available — model not saved (pip install joblib)")

    report_path = out / "prerun_router_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  saved report → {report_path}")

    print(f"\nPre-run router (TaskSpec-only) results:")
    print(f"  CV accuracy : {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
    print(f"  CV F1 macro : {result['cv_f1_macro_mean']:.4f} ± {result['cv_f1_macro_std']:.4f}")
    print(f"\nPost-attempt router (violation features) reference:")
    print(f"  CV accuracy : 0.9267 ± 0.0249")
    print(f"  CV F1 macro : 0.8589 ± 0.0531")
    acc_delta = result["cv_accuracy_mean"] - 0.9267
    f1_delta = result["cv_f1_macro_mean"] - 0.8589
    print(f"\nDelta (pre-run vs post-attempt):")
    print(f"  accuracy : {acc_delta:+.4f}")
    print(f"  F1 macro : {f1_delta:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
