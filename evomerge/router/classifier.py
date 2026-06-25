"""Rule-based router classifier for online use and baseline comparison.

The RouterRuleClassifier makes routing decisions purely from feature
thresholds — no ML model required.  It is the baseline that the eventual
ML classifier (GBDT / XGBoost / small transformer) must beat.

Decision logic mirrors plan Section 6 Phase 3, implemented as a
priority-ordered rule chain so the logic is easy to audit and adjust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from evomerge.router.features import RouterFeatures
from evomerge.router.labels import RouterLabel


@dataclass
class RouterConfig:
    """Threshold configuration for RouterRuleClassifier.

    Attributes:
        max_repair_rounds: if eval_repair_rounds exceeds this, escalate.
        max_violations: if violation_count exceeds this, escalate.
        min_tool_validity: if tool_validity_rate falls below this, escalate.
        max_latency_ms: if latency exceeds this, escalate (slow = likely complex).
        hard_constraint_limit: if n_hard exceeds this, route to large model directly.
    """
    max_repair_rounds: int = 3
    max_violations: int = 3
    min_tool_validity: float = 0.8
    max_latency_ms: float = 30_000.0
    hard_constraint_limit: int = 10


class RouterRuleClassifier:
    """Deterministic rule-based router.

    All decisions are fully explainable: each routing returns a label plus
    a human-readable reason string.

    Args:
        config: RouterConfig with threshold values.
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def predict(self, features: RouterFeatures) -> RouterLabel:
        """Return the routing label for the given feature vector."""
        return self._decide(features)[0]

    def predict_with_reason(self, features: RouterFeatures) -> tuple[RouterLabel, str]:
        """Return (label, reason) for explainability."""
        return self._decide(features)

    def _decide(self, f: RouterFeatures) -> tuple[RouterLabel, str]:
        cfg = self.config

        # --- hard escalation rules (order matters) ---
        if f.eval_escalated:
            return RouterLabel.need_large_model, "already escalated in prior round"

        if f.eval_repair_rounds > cfg.max_repair_rounds:
            return (
                RouterLabel.need_large_model,
                f"repair_rounds={f.eval_repair_rounds} > threshold={cfg.max_repair_rounds}",
            )

        if f.eval_violation_count > cfg.max_violations:
            return (
                RouterLabel.need_large_model,
                f"violation_count={f.eval_violation_count} > threshold={cfg.max_violations}",
            )

        if (
            f.eval_tool_calls_total > 0
            and f.eval_tool_validity_rate < cfg.min_tool_validity
        ):
            return (
                RouterLabel.need_repair,
                f"tool_validity_rate={f.eval_tool_validity_rate:.2f} < threshold={cfg.min_tool_validity}",
            )

        if f.taskspec_n_hard > cfg.hard_constraint_limit:
            return (
                RouterLabel.need_large_model,
                f"taskspec_n_hard={f.taskspec_n_hard} > limit={cfg.hard_constraint_limit}",
            )

        if f.eval_latency_ms > cfg.max_latency_ms:
            return (
                RouterLabel.need_large_model,
                f"latency={f.eval_latency_ms:.0f}ms > threshold={cfg.max_latency_ms:.0f}ms",
            )

        # --- soft repair hints ---
        if f.eval_repair_rounds > 0:
            return (
                RouterLabel.need_repair,
                f"repair_rounds={f.eval_repair_rounds} > 0; repair path active",
            )

        # --- default: small model is fine ---
        return RouterLabel.small_model_can_handle, "all thresholds satisfied"

    def predict_batch(
        self, features_list: list[RouterFeatures]
    ) -> list[RouterLabel]:
        """Batch predict over a list of feature vectors."""
        return [self.predict(f) for f in features_list]

    def evaluate(
        self,
        features_list: Sequence[RouterFeatures],
        labels: Sequence[RouterLabel],
    ) -> "RouterEvalReport":
        """Evaluate predictions against ground-truth labels.

        Args:
            features_list: feature vectors to predict on.
            labels: ground-truth RouterLabel for each vector.

        Returns:
            RouterEvalReport with accuracy, confusion matrix, and per-label
            failure buckets (false-positive and false-negative counts).
        """
        all_labels = list(RouterLabel)
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
        n = len(all_labels)
        matrix: list[list[int]] = [[0] * n for _ in range(n)]

        correct = 0
        for feat, true_lbl in zip(features_list, labels):
            pred_lbl = self.predict(feat)
            matrix[label_to_idx[true_lbl]][label_to_idx[pred_lbl]] += 1
            if pred_lbl == true_lbl:
                correct += 1

        total = len(labels)
        accuracy = correct / total if total else 0.0

        # Per-label: false positives (predicted this, not actually this)
        # and false negatives (actually this, not predicted this)
        failure_buckets: dict[str, dict[str, int]] = {}
        for i, lbl in enumerate(all_labels):
            tp = matrix[i][i]
            fp = sum(matrix[j][i] for j in range(n) if j != i)
            fn = sum(matrix[i][j] for j in range(n) if j != i)
            failure_buckets[lbl.value] = {"tp": tp, "fp": fp, "fn": fn}

        return RouterEvalReport(
            n_total=total,
            accuracy=accuracy,
            labels=[lbl.value for lbl in all_labels],
            confusion_matrix=matrix,
            failure_buckets=failure_buckets,
        )


@dataclass
class RouterEvalReport:
    """Evaluation report for RouterRuleClassifier.predict_batch against ground truth."""

    n_total: int
    accuracy: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    failure_buckets: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "accuracy": round(self.accuracy, 4),
            "labels": self.labels,
            "confusion_matrix": self.confusion_matrix,
            "failure_buckets": self.failure_buckets,
        }

    def print_summary(self) -> None:
        print(f"Accuracy: {self.accuracy:.1%}  (n={self.n_total})")
        print("\nConfusion matrix (rows=true, cols=predicted):")
        width = max(len(lbl) for lbl in self.labels)
        header = " " * (width + 2) + "  ".join(f"{lbl[:8]:>8}" for lbl in self.labels)
        print(header)
        for i, lbl in enumerate(self.labels):
            row = "  ".join(f"{self.confusion_matrix[i][j]:>8}" for j in range(len(self.labels)))
            print(f"  {lbl:{width}}  {row}")
        print("\nFailure buckets (tp / fp / fn per label):")
        for lbl, counts in self.failure_buckets.items():
            print(f"  {lbl:30}  tp={counts['tp']:4}  fp={counts['fp']:4}  fn={counts['fn']:4}")


__all__ = ["RouterConfig", "RouterRuleClassifier", "RouterEvalReport"]
