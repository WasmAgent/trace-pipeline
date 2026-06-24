"""evomerge — WasmAgent trace-to-training pipeline and LLM benchmark audit toolkit.

Public API surface:

Training data pipeline:
    evomerge.schemas      — Pydantic models (RolloutBranchRecord, ComplianceEvalRecord, training records)
    evomerge.pipeline     — trace → SFT / DPO / PPO converters
    evomerge.io           — JSONL load / write helpers
    evomerge.export       — run_export() full pipeline
    evomerge.validate     — contamination and schema checks
    evomerge.synthesize   — synthetic sample generation via teacher model
    evomerge.router       — routing feature extraction, labels, rule classifier

Evaluation harness:
    evomerge.eval         — EvalHarness, EvalMetrics, stat_bridge (McNemar / bootstrap)

Benchmark audit (eval_trust):
    eval_trust.paired_stats  — McNemar, Wilson CI, paired bootstrap
    eval_trust.conformal_ci  — distribution-free prediction intervals
    eval_trust.lm_eval_bridge— lm-evaluation-harness adapter
    eval_trust.t0v2          — T0v2 channel audit (truncation, aggregate)
"""
from evomerge.export import ExportManifest, run_export
from evomerge.io import (
    load_compliance_records,
    load_jsonl,
    load_rollouts,
    load_router_records,
    write_dicts_jsonl,
    write_jsonl,
)
from evomerge.schemas import (
    ComplianceEvalRecord,
    ConstraintViolation,
    DpoTrainingRecord,
    Message,
    PpoTrainingRecord,
    Provenance,
    RepairTraceEntry,
    RolloutBranchRecord,
    SftTrainingRecord,
    TaskSpec,
)

__all__ = [
    # export
    "ExportManifest",
    "run_export",
    # io
    "load_compliance_records",
    "load_jsonl",
    "load_rollouts",
    "load_router_records",
    "write_dicts_jsonl",
    "write_jsonl",
    # schemas
    "ComplianceEvalRecord",
    "ConstraintViolation",
    "DpoTrainingRecord",
    "Message",
    "PpoTrainingRecord",
    "Provenance",
    "RepairTraceEntry",
    "RolloutBranchRecord",
    "SftTrainingRecord",
    "TaskSpec",
]
