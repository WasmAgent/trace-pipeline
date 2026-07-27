"""Core metric types and computation for the compliance eval harness.

Metrics (plan Section 7.4):
  taskspec_pass_rate       — fraction of runs where final_pass=True
  tool_call_validity       — fraction of tool calls that match declared schema
  repair_success_rate      — fraction of repair rounds that resolve all violations
  evidence_sufficiency     — fraction of outputs with sufficient evidence citations
  fallback_rate            — fraction of tasks that escalated to large model / human
  avg_repair_rounds        — mean number of repair rounds across all runs
  cost_per_accepted_task   — mean total token cost for accepted (final_pass=True) runs
  latency_per_accepted_ms  — mean latency for accepted runs
  general_ability_score    — optional hold-out benchmark score (set externally)
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class EvalRecord:
    """One evaluated run result, group-agnostic.

    Attributes:
        task_id: task identifier.
        group: group label (A–E).
        final_pass: whether the output satisfied all constraints.
        tool_calls_total: number of tool calls made.
        tool_calls_valid: number of tool calls with valid schema/args.
        repair_rounds: number of repair rounds attempted.
        repair_rounds_ok: number of repair rounds that fully resolved violations.
        has_evidence: True if the output contains sufficient evidence citations.
        escalated: True if the task was escalated to a larger model or human.
        prompt_tokens: total prompt tokens used.
        generation_tokens: total generation tokens used.
        repair_tokens: tokens used in repair rounds.
        latency_ms: wall-clock latency in milliseconds.
        general_score: optional hold-out benchmark score (0–1).
    """
    task_id: str
    group: str
    final_pass: bool
    tool_calls_total: int = 0
    tool_calls_valid: int = 0
    repair_rounds: int = 0
    repair_rounds_ok: int = 0
    has_evidence: bool = True
    escalated: bool = False
    prompt_tokens: int = 0
    generation_tokens: int = 0
    repair_tokens: int = 0
    latency_ms: float = 0.0
    general_score: float | None = None


@dataclass
class EvalMetrics:
    """Aggregate metrics for one experiment group."""
    group: str
    n: int

    taskspec_pass_rate: float = 0.0
    tool_call_validity: float = 0.0
    repair_success_rate: float = 0.0
    evidence_sufficiency: float = 0.0
    fallback_rate: float = 0.0
    avg_repair_rounds: float = 0.0
    cost_per_accepted_task: float = 0.0
    latency_per_accepted_ms: float = 0.0
    general_ability_score: float | None = None

    # 95% Wilson CIs on binary rates (lo, hi)
    taskspec_pass_ci: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    tool_call_validity_ci: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "n": self.n,
            "taskspec_pass_rate": round(self.taskspec_pass_rate, 4),
            "taskspec_pass_ci": [round(v, 4) for v in self.taskspec_pass_ci],
            "tool_call_validity": round(self.tool_call_validity, 4),
            "tool_call_validity_ci": [round(v, 4) for v in self.tool_call_validity_ci],
            "repair_success_rate": round(self.repair_success_rate, 4),
            "evidence_sufficiency": round(self.evidence_sufficiency, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "avg_repair_rounds": round(self.avg_repair_rounds, 3),
            "cost_per_accepted_task": round(self.cost_per_accepted_task, 1),
            "latency_per_accepted_ms": round(self.latency_per_accepted_ms, 1),
            "general_ability_score": (
                round(self.general_ability_score, 4)
                if self.general_ability_score is not None
                else None
            ),
        }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_metrics(records: Sequence[EvalRecord]) -> EvalMetrics:
    """Compute aggregate EvalMetrics from a list of EvalRecord for one group.

    All records must belong to the same group.
    """
    if not records:
        raise ValueError("records must not be empty")
    group = records[0].group
    n = len(records)

    n_pass = sum(1 for r in records if r.final_pass)
    n_pass_ci = _wilson_ci(n_pass, n)

    # tool call validity
    total_calls = sum(r.tool_calls_total for r in records)
    valid_calls = sum(r.tool_calls_valid for r in records)
    tc_validity = valid_calls / total_calls if total_calls > 0 else 1.0
    tc_ci = _wilson_ci(valid_calls, total_calls)

    # repair success rate
    total_repair_rounds = sum(r.repair_rounds for r in records)
    ok_repair_rounds = sum(r.repair_rounds_ok for r in records)
    repair_success = ok_repair_rounds / total_repair_rounds if total_repair_rounds > 0 else 1.0

    evidence_suf = sum(1 for r in records if r.has_evidence) / n
    fallback_rate = sum(1 for r in records if r.escalated) / n
    avg_repair = sum(r.repair_rounds for r in records) / n

    accepted = [r for r in records if r.final_pass]
    if accepted:
        cost_per = sum(
            r.prompt_tokens + r.generation_tokens + r.repair_tokens for r in accepted
        ) / len(accepted)
        latency_per = sum(r.latency_ms for r in accepted) / len(accepted)
    else:
        cost_per = 0.0
        latency_per = 0.0

    general_scores = [r.general_score for r in records if r.general_score is not None]
    general_ability = sum(general_scores) / len(general_scores) if general_scores else None

    return EvalMetrics(
        group=group,
        n=n,
        taskspec_pass_rate=n_pass / n,
        taskspec_pass_ci=n_pass_ci,
        tool_call_validity=tc_validity,
        tool_call_validity_ci=tc_ci,
        repair_success_rate=repair_success,
        evidence_sufficiency=evidence_suf,
        fallback_rate=fallback_rate,
        avg_repair_rounds=avg_repair,
        cost_per_accepted_task=cost_per,
        latency_per_accepted_ms=latency_per,
        general_ability_score=general_ability,
    )


__all__ = ["EvalMetrics", "EvalRecord", "compute_metrics"]
