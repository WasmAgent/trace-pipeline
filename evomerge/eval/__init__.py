"""evomerge.eval — evaluation harness for the A/B/C/D/E comparison experiment.

Experiment groups (plan Section 7.2):
  A: Base small model, direct prompt
  B: Base small model + compliance-engine
  C: Fine-tuned small model + compliance-engine
  D: Large model, direct prompt
  E: Large model + compliance-engine (optional)

This module provides:
  - EvalRecord: one evaluated result row
  - EvalMetrics: aggregate metrics for one group
  - EvalHarness: runs all groups and computes metrics
  - paired_significance / compare_all_groups: McNemar + bootstrap significance
"""
from evomerge.eval.metrics import EvalMetrics, EvalRecord, compute_metrics
from evomerge.eval.harness import EvalConfig, EvalGroup, EvalHarness, EvalReport
from evomerge.eval.stat_bridge import (
    SignificanceReport,
    compare_all_groups,
    paired_significance,
)

__all__ = [
    "EvalConfig",
    "EvalGroup",
    "EvalHarness",
    "EvalMetrics",
    "EvalRecord",
    "EvalReport",
    "SignificanceReport",
    "compare_all_groups",
    "compute_metrics",
    "paired_significance",
]
