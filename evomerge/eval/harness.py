"""EvalHarness — orchestrates the A/B/C/D/E comparison experiment.

The harness is model-agnostic: callers supply a `run_fn` per group that
takes a task description and returns an EvalRecord.  The harness handles
batching, result collection, metric computation, and JSONL export.

Typical usage:

    from evomerge.eval import EvalConfig, EvalGroup, EvalHarness, EvalRecord

    def run_group_a(task_id: str, task: str) -> EvalRecord:
        # call base model directly, no compliance engine
        output = base_model.generate(task)
        return EvalRecord(
            task_id=task_id, group="A",
            final_pass=verifier.check(output),
            ...
        )

    harness = EvalHarness(
        config=EvalConfig(task_ids=task_ids, tasks=tasks),
        groups={
            "A": EvalGroup(label="A", run_fn=run_group_a),
            "C": EvalGroup(label="C", run_fn=run_group_c),
        },
    )
    report = harness.run()
    harness.save(report, "results/eval_report.json")
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evomerge.eval.metrics import EvalMetrics, EvalRecord, compute_metrics

RunFn = Callable[[str, str], EvalRecord]


@dataclass
class EvalGroup:
    """One comparison group.

    Attributes:
        label: group label (e.g. "A", "B", "C", "D", "E").
        run_fn: callable(task_id, task_text) -> EvalRecord.
        description: human-readable description for the report.
    """
    label: str
    run_fn: RunFn
    description: str = ""


@dataclass
class EvalConfig:
    """Configuration shared across all groups.

    Attributes:
        task_ids: list of task identifiers (parallel with tasks).
        tasks: list of task prompt strings.
        stop_on_error: if True, raise on run_fn exceptions; else log and skip.
    """
    task_ids: list[str]
    tasks: list[str]
    stop_on_error: bool = False

    def __post_init__(self):
        if len(self.task_ids) != len(self.tasks):
            raise ValueError("task_ids and tasks must have the same length")


@dataclass
class EvalReport:
    """Full comparison report across all groups."""
    metrics: dict[str, EvalMetrics]
    errors: dict[str, list[str]] = field(default_factory=dict)

    def summary_table(self) -> list[dict]:
        return [m.to_dict() for m in self.metrics.values()]

    def to_dict(self) -> dict:
        return {
            "metrics": {g: m.to_dict() for g, m in self.metrics.items()},
            "errors": self.errors,
        }


class EvalHarness:
    """Run the A/B/C/D/E experiment and collect metrics.

    Args:
        config: EvalConfig with task list.
        groups: dict mapping group label to EvalGroup.
    """

    def __init__(self, config: EvalConfig, groups: dict[str, EvalGroup]):
        self.config = config
        self.groups = groups

    def run(self) -> EvalReport:
        """Execute all groups over all tasks. Returns EvalReport."""
        all_records: dict[str, list[EvalRecord]] = {g: [] for g in self.groups}
        all_errors: dict[str, list[str]] = {g: [] for g in self.groups}

        for task_id, task in zip(self.config.task_ids, self.config.tasks, strict=False):
            for label, grp in self.groups.items():
                try:
                    rec = grp.run_fn(task_id, task)
                    all_records[label].append(rec)
                except Exception as exc:
                    msg = f"{task_id}: {exc}"
                    all_errors[label].append(msg)
                    if self.config.stop_on_error:
                        raise

        metrics: dict[str, EvalMetrics] = {}
        for label, records in all_records.items():
            if records:
                metrics[label] = compute_metrics(records)

        return EvalReport(metrics=metrics, errors=all_errors)

    def save(self, report: EvalReport, path: str | Path) -> None:
        """Write the report to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        )

    def save_records_jsonl(
        self,
        records: Sequence[EvalRecord],
        path: str | Path,
    ) -> None:
        """Write raw EvalRecord list to JSONL for downstream analysis."""
        from dataclasses import asdict

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


__all__ = ["EvalConfig", "EvalGroup", "EvalHarness", "EvalReport"]
