"""Router label generation and RouterRecord (feature, label) pairs.

RouterLabel values (plan Section 6 Phase 3 outputs):
  small_model_can_handle  — small model succeeded; no escalation needed
  need_repair             — small model failed; repair is worth trying
  need_large_model        — repair failed or task is too complex for small model
  need_human_review       — high-stakes or repeated failure; human should review

Label is derived from the outcome of a completed EvalRecord.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from evomerge.eval.metrics import EvalRecord
from evomerge.router.features import RouterFeatures, feature_from_record
from evomerge.schemas.compliance import TaskSpec
from evomerge.schemas.training import Provenance


class RouterLabel(str, Enum):
    small_model_can_handle = "small_model_can_handle"
    need_repair = "need_repair"
    need_large_model = "need_large_model"
    need_human_review = "need_human_review"


@dataclass
class RouterRecord:
    """(features, label) pair for router classifier training."""
    task_id: str
    features: RouterFeatures
    label: RouterLabel
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "features": self.features.to_dict(),
            "label": self.label.value,
            "provenance": {
                "source": self.provenance.source,
                "task_id": self.provenance.task_id,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> RouterRecord:
        from dataclasses import fields as dc_fields
        feat_data = d["features"]
        feat = RouterFeatures(**{f.name: feat_data[f.name] for f in dc_fields(RouterFeatures)})
        prov_data = d.get("provenance", {})
        prov = Provenance(
            source=prov_data.get("source", ""),
            task_id=prov_data.get("task_id"),
        )
        return cls(
            task_id=d["task_id"],
            features=feat,
            label=RouterLabel(d["label"]),
            provenance=prov,
        )


def label_from_record(
    record: EvalRecord,
    *,
    max_repair_before_escalate: int = 3,
    escalate_on_repeated_failure: int = 2,
) -> RouterLabel:
    """Derive a RouterLabel from a completed EvalRecord.

    Decision rules (deterministic, mirrors plan Section 6):

      1. escalated=True  OR  repair_rounds > max_repair_before_escalate
            → need_large_model
      2. final_pass=True AND repair_rounds == 0
            → small_model_can_handle
      3. final_pass=True AND repair_rounds > 0
            → need_repair  (repair was needed but succeeded; train repair path)
      4. final_pass=False AND repair_rounds < escalate_on_repeated_failure
            → need_repair  (still worth trying repair)
      5. final_pass=False AND repair_rounds >= escalate_on_repeated_failure
            → need_large_model

    Args:
        record: a completed EvalRecord.
        max_repair_before_escalate: repair_rounds above this → escalate.
        escalate_on_repeated_failure: failed runs with this many rounds → escalate.

    Returns:
        RouterLabel.
    """
    if record.escalated or record.repair_rounds > max_repair_before_escalate:
        return RouterLabel.need_large_model

    if record.final_pass and record.repair_rounds == 0:
        return RouterLabel.small_model_can_handle

    if record.final_pass:
        return RouterLabel.need_repair

    if record.repair_rounds < escalate_on_repeated_failure:
        return RouterLabel.need_repair

    return RouterLabel.need_large_model


def build_router_records(
    specs: dict[str, TaskSpec],
    records: Sequence[EvalRecord],
    *,
    source: str = "wasmagent-eval",
) -> list[RouterRecord]:
    """Build RouterRecord list from TaskSpec map and EvalRecord list.

    Args:
        specs: dict mapping task_id → TaskSpec.
        records: EvalRecord list (typically from the small-model group).
        source: provenance source label.

    Returns:
        List of RouterRecord, one per record that has a matching TaskSpec.
    """
    results: list[RouterRecord] = []
    for rec in records:
        spec = specs.get(rec.task_id)
        if spec is None:
            continue
        features = feature_from_record(spec, rec)
        label = label_from_record(rec)
        results.append(
            RouterRecord(
                task_id=rec.task_id,
                features=features,
                label=label,
                provenance=Provenance(source=source, task_id=rec.task_id),
            )
        )
    return results


__all__ = ["RouterLabel", "RouterRecord", "build_router_records", "label_from_record"]
