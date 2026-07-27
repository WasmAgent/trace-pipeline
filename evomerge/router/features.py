"""Router feature extraction from TaskSpec + EvalRecord.

Feature vector for the router / escalation classifier (plan Section 6):

  taskspec_n_constraints    — total number of constraints
  taskspec_n_hard           — number of hard constraints
  taskspec_has_tools        — whether the task requires tool calls
  taskspec_n_allowed_tools  — number of allowed tools
  taskspec_max_repair_rounds— max_rounds from TaskSpecRepairConfig
  eval_repair_rounds        — actual repair rounds used (0 if no EvalRecord)
  eval_violation_count      — number of violations detected
  eval_hard_violation_count — number of hard-level violations
  eval_tool_calls_total     — total tool calls made
  eval_tool_calls_valid     — valid tool calls
  eval_tool_validity_rate   — valid / total (1.0 if no calls)
  eval_escalated            — 1 if already escalated in this run, 0 otherwise
  eval_latency_ms           — wall-clock latency
  eval_prompt_tokens        — prompt tokens used
  eval_generation_tokens    — generation tokens used
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from evomerge.eval.metrics import EvalRecord
from evomerge.schemas.compliance import ConstraintLevel, TaskSpec


@dataclass
class RouterFeatures:
    """Structured feature vector for the router model."""
    taskspec_n_constraints: int
    taskspec_n_hard: int
    taskspec_has_tools: int          # 0 or 1
    taskspec_n_allowed_tools: int
    taskspec_max_repair_rounds: int
    eval_repair_rounds: int
    eval_violation_count: int
    eval_hard_violation_count: int
    eval_tool_calls_total: int
    eval_tool_calls_valid: int
    eval_tool_validity_rate: float
    eval_escalated: int              # 0 or 1
    eval_latency_ms: float
    eval_prompt_tokens: int
    eval_generation_tokens: int

    def to_list(self) -> list[float]:
        """Flat numeric feature vector suitable for sklearn / XGBoost."""
        return [float(v) for v in asdict(self).values()]

    def to_dict(self) -> dict:
        return asdict(self)


def feature_from_record(
    spec: TaskSpec,
    record: EvalRecord | None = None,
) -> RouterFeatures:
    """Extract RouterFeatures from a TaskSpec and optional EvalRecord.

    When record is None (pre-run routing), eval_* fields default to 0.

    Args:
        spec: the TaskSpec for the current task.
        record: EvalRecord from a prior small-model attempt (can be None).

    Returns:
        RouterFeatures instance.
    """
    n_constraints = len(spec.constraints)
    n_hard = sum(1 for c in spec.constraints if c.level == ConstraintLevel.hard)
    has_tools = int(spec.tools is not None and len(spec.tools.allowed) > 0)
    n_tools = len(spec.tools.allowed) if spec.tools else 0
    max_repair = spec.repair.max_rounds if spec.repair else 3

    if record is not None:
        tc_total = record.tool_calls_total
        tc_valid = record.tool_calls_valid
        tc_rate = tc_valid / tc_total if tc_total > 0 else 1.0
        return RouterFeatures(
            taskspec_n_constraints=n_constraints,
            taskspec_n_hard=n_hard,
            taskspec_has_tools=has_tools,
            taskspec_n_allowed_tools=n_tools,
            taskspec_max_repair_rounds=max_repair,
            eval_repair_rounds=record.repair_rounds,
            eval_violation_count=record.repair_rounds,   # proxy: rounds ≈ violations found
            eval_hard_violation_count=0,                 # not tracked in EvalRecord; override if available
            eval_tool_calls_total=tc_total,
            eval_tool_calls_valid=tc_valid,
            eval_tool_validity_rate=tc_rate,
            eval_escalated=int(record.escalated),
            eval_latency_ms=record.latency_ms,
            eval_prompt_tokens=record.prompt_tokens,
            eval_generation_tokens=record.generation_tokens,
        )

    return RouterFeatures(
        taskspec_n_constraints=n_constraints,
        taskspec_n_hard=n_hard,
        taskspec_has_tools=has_tools,
        taskspec_n_allowed_tools=n_tools,
        taskspec_max_repair_rounds=max_repair,
        eval_repair_rounds=0,
        eval_violation_count=0,
        eval_hard_violation_count=0,
        eval_tool_calls_total=0,
        eval_tool_calls_valid=0,
        eval_tool_validity_rate=1.0,
        eval_escalated=0,
        eval_latency_ms=0.0,
        eval_prompt_tokens=0,
        eval_generation_tokens=0,
    )


__all__ = ["RouterFeatures", "feature_from_record"]
