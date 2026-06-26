"""evomerge.export — end-to-end pipeline from raw traces to training JSONL.

Typical usage:

    from evomerge.export import run_export

    manifest = run_export(
        rollout_jsonl="data/rollouts.jsonl",
        compliance_jsonl="data/compliance.jsonl",  # optional
        out_dir="data/training/",
        eval_texts=eval_items,                     # optional, for contamination
    )
    print(manifest)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from evomerge.io import (
    load_compliance_records,
    load_rollouts,
    write_dicts_jsonl,
    write_jsonl,
)
from evomerge.pipeline import (
    compliance_to_dpo_records,
    compliance_to_sft_records,
    to_dpo_records,
    to_ppo_records,
    to_sft_records,
)
from evomerge.validate import check_contamination, validate_training_record
from evomerge.validate.redaction import (
    RedactionReport,
    BSCODE_REDACTED_FIELDS,
    BSCODE_PATTERNS,
)
from evomerge.dataset_card import generate_dataset_card


@dataclass
class ExportManifest:
    out_dir: str
    n_sft: int = 0
    n_dpo: int = 0
    n_ppo: int = 0
    n_compliance_sft: int = 0
    n_compliance_dpo: int = 0
    n_router: int = 0
    n_invalid: int = 0
    n_contaminated: int = 0
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "n_sft": self.n_sft,
            "n_dpo": self.n_dpo,
            "n_ppo": self.n_ppo,
            "n_compliance_sft": self.n_compliance_sft,
            "n_compliance_dpo": self.n_compliance_dpo,
            "n_router": self.n_router,
            "n_invalid": self.n_invalid,
            "n_contaminated": self.n_contaminated,
            "files": self.files,
        }


def run_export(
    *,
    rollout_jsonl: str | Path | None = None,
    compliance_jsonl: str | Path | None = None,
    out_dir: str | Path = "data/training",
    eval_texts: Sequence[str] | None = None,
    contamination_threshold: float = 0.2,
    only_passing_sft: bool = True,
    task_specs: dict | None = None,
    eval_records: list | None = None,
    router_source: str = "wasmagent-eval",
    require_attested_dpo: bool = False,
) -> ExportManifest:
    """Full pipeline: load traces, convert, validate, decontaminate, export.

    Args:
        rollout_jsonl: path to rollout-wire/v1 JSONL from wasmagent-js.
        compliance_jsonl: path to ComplianceEvalRecord JSONL (optional).
        out_dir: directory to write training JSONL files.
        eval_texts: eval item texts for contamination check (optional).
        contamination_threshold: Jaccard threshold for flagging records.
        only_passing_sft: if True, only include objective_score=1 branches
            in rollout SFT records.
        task_specs: dict[task_id → TaskSpec] for router record generation.
            When provided together with eval_records, router.jsonl is written.
        eval_records: list[EvalRecord] from the small-model group.
        router_source: provenance source label for RouterRecord.
        require_attested_dpo: when True, DPO pairs are only produced from
            branches whose verifier_results contain an entry with
            evidence_source == "attested" and a non-empty signer.key_id.
            Defaults to False for backward compatibility with fixtures that
            predate the attestation requirement.  Set to True in production
            pipelines that consume wasmagent-js AEP v0.2+ rollout data.

    Returns:
        ExportManifest with counts and output file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = ExportManifest(out_dir=str(out))

    rollouts = load_rollouts(rollout_jsonl) if rollout_jsonl else []
    compliance = (
        load_compliance_records(compliance_jsonl) if compliance_jsonl else []
    )

    sft_records = to_sft_records(rollouts, only_passing=only_passing_sft)
    dpo_records = to_dpo_records(rollouts, require_attested_evidence=require_attested_dpo)
    ppo_records = to_ppo_records(rollouts)
    compliance_sft = compliance_to_sft_records(compliance)

    # --- validation ---
    n_invalid = 0
    for rec_list in (sft_records, dpo_records, ppo_records, compliance_sft):
        for rec in rec_list:
            result = validate_training_record(rec)
            if not result.ok:
                n_invalid += 1
    manifest.n_invalid = n_invalid

    # --- contamination ---
    if eval_texts:
        all_outputs = (
            [r.messages[-1].content for r in sft_records]
            + [r.chosen for r in dpo_records]
            + [r.messages[-1].content for r in ppo_records]
            + [r.messages[-1].content for r in compliance_sft]
        )
        report = check_contamination(
            all_outputs, list(eval_texts), threshold=contamination_threshold
        )
        manifest.n_contaminated = report.n_flagged

    # --- write training files ---
    if sft_records:
        p = out / "sft.jsonl"
        write_jsonl(sft_records, p)
        manifest.n_sft = len(sft_records)
        manifest.files["sft"] = str(p)

    if dpo_records:
        p = out / "dpo.jsonl"
        write_jsonl(dpo_records, p)
        manifest.n_dpo = len(dpo_records)
        manifest.files["dpo"] = str(p)

    if ppo_records:
        p = out / "ppo.jsonl"
        write_jsonl(ppo_records, p)
        manifest.n_ppo = len(ppo_records)
        manifest.files["ppo"] = str(p)

    if compliance_sft:
        p = out / "compliance_sft.jsonl"
        write_jsonl(compliance_sft, p)
        manifest.n_compliance_sft = len(compliance_sft)
        manifest.files["compliance_sft"] = str(p)

    # --- compliance DPO pairs ---
    compliance_dpo = compliance_to_dpo_records(compliance)
    if compliance_dpo:
        p = out / "compliance_dpo.jsonl"
        write_jsonl(compliance_dpo, p)
        manifest.n_compliance_dpo = len(compliance_dpo)
        manifest.files["compliance_dpo"] = str(p)

    # --- router records ---
    if task_specs and eval_records:
        from evomerge.router.labels import build_router_records
        router_records = build_router_records(
            task_specs, eval_records, source=router_source
        )
        if router_records:
            p = out / "router.jsonl"
            write_dicts_jsonl([r.to_dict() for r in router_records], p)
            manifest.n_router = len(router_records)
            manifest.files["router"] = str(p)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    )
    manifest.files["manifest"] = str(manifest_path)

    # --- contamination report (always written, even when no eval_texts) ---
    contamination_report: dict = {
        "n_training": (
            manifest.n_sft + manifest.n_dpo + manifest.n_ppo
            + manifest.n_compliance_sft + manifest.n_compliance_dpo
        ),
        "n_contaminated": manifest.n_contaminated,
        "contamination_threshold": contamination_threshold,
        "eval_texts_provided": eval_texts is not None,
        "flag_rate": (
            manifest.n_contaminated
            / max(manifest.n_sft + manifest.n_dpo + manifest.n_ppo
                  + manifest.n_compliance_sft + manifest.n_compliance_dpo, 1)
        ),
    }
    contamination_path = out / "contamination_report.json"
    contamination_path.write_text(
        json.dumps(contamination_report, indent=2, ensure_ascii=False)
    )
    manifest.files["contamination_report"] = str(contamination_path)

    # --- schema report (record counts + file sizes by type) ---
    schema_report: dict = {
        "schema_versions": {
            "rollout_wire": "rollout-wire/v1",
            "training_record": "training-record/v1",
        },
        "record_counts": {
            "sft": manifest.n_sft,
            "dpo": manifest.n_dpo,
            "ppo": manifest.n_ppo,
            "compliance_sft": manifest.n_compliance_sft,
            "compliance_dpo": manifest.n_compliance_dpo,
            "router": manifest.n_router,
        },
        "n_invalid": manifest.n_invalid,
        "output_files": {
            k: {"path": v, "size_bytes": Path(v).stat().st_size if Path(v).exists() else 0}
            for k, v in manifest.files.items()
            if k not in ("schema_report",)
        },
    }
    schema_path = out / "schema_report.json"
    schema_path.write_text(
        json.dumps(schema_report, indent=2, ensure_ascii=False)
    )
    manifest.files["schema_report"] = str(schema_path)

    # --- redaction report (describes what PII checks were applied to source data) ---
    # The source redaction version is read from the first record's provenance when
    # available; otherwise defaults to the bscode standard.
    redaction_version = "bscode/pii-redact/v1"
    evidence_source = "client_reported"
    if rollouts:
        prov = getattr(rollouts[0], "provenance", None)
        if isinstance(prov, dict):
            redaction_version = prov.get("redaction_version", redaction_version)
            evidence_source = prov.get("evidence_source", evidence_source)

    redaction_report = RedactionReport(
        redaction_version=redaction_version,
        evidence_source=evidence_source,
        fields_redacted=BSCODE_REDACTED_FIELDS,
        patterns_applied=BSCODE_PATTERNS,
        n_records_scanned=len(rollouts),
    )
    redaction_path = out / "redaction_report.json"
    redaction_path.write_text(
        json.dumps(redaction_report.to_dict(), indent=2, ensure_ascii=False)
    )
    manifest.files["redaction_report"] = str(redaction_path)

    # --- dataset card (auto-generated; name defaults to out_dir basename) ---
    dataset_name = Path(out_dir).name or "wasmagent-dataset"
    card_md = generate_dataset_card(manifest.to_dict(), name=dataset_name)
    card_path = out / "DATASET_CARD.md"
    card_path.write_text(card_md)
    manifest.files["dataset_card"] = str(card_path)

    # Rewrite manifest now that contamination_report + schema_report paths are known
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    )

    return manifest

__all__ = ["ExportManifest", "run_export"]
