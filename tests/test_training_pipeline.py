"""Tests for evomerge.training_pipeline — Milestone 5 training-pipeline integration.

Covers the four sub-requirements of issue #54 (DPO/PPO/SFT training pipeline
integration) each in its own test class:

  - TestDatasetVersioning      — dataset versioning (content-addressed)
  - TestCheckpointRegistry     — checkpoint management (lineage)
  - TestLossTelemetry          — training-loss telemetry → trust-score feedback
  - TestTrainingJobManifest    — wire exporters → orchestration manifest
"""
from __future__ import annotations

import json

import pytest

from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import Message, Provenance, SftTrainingRecord
from evomerge.training_pipeline import (
    ORCHESTRATORS,
    TRAINING_MODES,
    CheckpointEntry,
    CheckpointRegistry,
    DatasetVersion,
    LossTelemetry,
    TrainingJobManifest,
    build_training_job_manifest,
    compute_dataset_digest,
    load_dataset_manifest,
    parse_trainer_log,
    records_for_mode,
    telemetry_from_loss_history,
    telemetry_from_summary,
    telemetry_to_training_health,
    version_dataset,
)
from evomerge.trust_score import AgentTrustScoreBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sft_dicts(n: int = 2) -> list[dict]:
    return [
        {
            "schema_version": "sft/v1",
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
            "output_type": "final_answer",
            "provenance": {"source": "test"},
        }
        for i in range(n)
    ]


def _sft_model() -> SftTrainingRecord:
    return SftTrainingRecord(
        messages=[Message(role="user", content="hi"), Message(role="assistant", content="yo")],
        output_type="final_answer",
        provenance=Provenance(source="test"),
    )


def _branch(rollout_id="r1", branch_index=0, score=1, status="pass", answer="Good."):
    return RolloutBranchRecord(
        rollout_id=rollout_id,
        task="Do the thing.",
        branch_index=branch_index,
        temperature=0.7,
        session_id="s1",
        final_answer=answer,
        objective_score=score,
        objective_status=status,
        rank=branch_index,
        total_score=float(score),
    )


# ===========================================================================
# 1. Dataset versioning
# ===========================================================================

class TestDatasetVersioning:
    def test_version_writes_jsonl_and_manifest(self, tmp_path):
        dv = version_dataset(_sft_dicts(3), name="sft", out_dir=tmp_path, sources=["a.jsonl"])
        assert dv.name == "sft"
        assert dv.schema_version == "sft/v1"
        assert dv.record_count == 3
        assert dv.version.startswith("v-")
        assert dv.sources == ["a.jsonl"]
        # files exist
        assert (tmp_path / f"sft-{dv.version}.jsonl").exists()
        assert (tmp_path / f"sft-{dv.version}.manifest.json").exists()

    def test_content_addressed_idempotent(self, tmp_path):
        records = _sft_dicts(2)
        dv1 = version_dataset(records, name="sft", out_dir=tmp_path)
        dv2 = version_dataset(records, name="sft", out_dir=tmp_path)
        assert dv1.content_digest == dv2.content_digest
        assert dv1.version == dv2.version
        assert dv1.path == dv2.path  # same content → same versioned path

    def test_different_content_different_version(self, tmp_path):
        dv1 = version_dataset(_sft_dicts(2), name="sft", out_dir=tmp_path)
        dv2 = version_dataset(_sft_dicts(3), name="sft", out_dir=tmp_path)
        assert dv1.content_digest != dv2.content_digest
        assert dv1.version != dv2.version

    def test_digest_key_order_independent(self):
        a = {"schema_version": "sft/v1", "messages": [], "a": 1, "b": 2}
        b = {"b": 2, "a": 1, "messages": [], "schema_version": "sft/v1"}
        assert compute_dataset_digest([a]) == compute_dataset_digest([b])

    def test_manifest_round_trip(self, tmp_path):
        dv = version_dataset(_sft_dicts(1), name="dpo", out_dir=tmp_path)
        loaded = load_dataset_manifest(dv.manifest_path)
        assert isinstance(loaded, DatasetVersion)
        assert loaded == dv

    def test_accepts_pydantic_models(self, tmp_path):
        dv = version_dataset([_sft_model()], name="sft", out_dir=tmp_path)
        assert dv.record_count == 1
        assert dv.schema_version == "sft/v1"

    def test_rejects_unknown_mode(self, tmp_path):
        with pytest.raises(ValueError, match="unknown training mode"):
            version_dataset(_sft_dicts(1), name="grpo", out_dir=tmp_path)


# ===========================================================================
# 2. Checkpoint management
# ===========================================================================

class TestCheckpointRegistry:
    def _entry(self, ckpt_id, mode="sft", base_ref="Qwen/Qwen2.5-1.5B-Instruct", **kw):
        return CheckpointEntry(
            checkpoint_id=ckpt_id,
            mode=mode,
            dataset_version=kw.get("dataset_version", "digest-a"),
            base_ref=base_ref,
            status=kw.get("status", "ready"),
            metrics=kw.get("metrics", {}),
        )

    def test_register_and_find(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        e = reg.register(self._entry("sft-v1"))
        assert reg.find("sft-v1") == e
        assert reg.find("missing") is None

    def test_register_dedup(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        reg.register(self._entry("sft-v1"))
        reg.register(self._entry("sft-v1", status="failed"))
        assert len(reg.all()) == 1
        # first write wins
        assert reg.find("sft-v1").status == "ready"

    def test_latest_by_mode_and_time(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        reg.register(CheckpointEntry("sft-1", "sft", "d1", "base", created_at_ms=100))
        reg.register(CheckpointEntry("dpo-1", "dpo", "d2", "sft-1", created_at_ms=200))
        reg.register(CheckpointEntry("sft-2", "sft", "d3", "base", created_at_ms=300))
        assert reg.latest().checkpoint_id == "sft-2"
        assert reg.latest(mode="dpo").checkpoint_id == "dpo-1"
        assert reg.latest(mode="ppo") is None

    def test_lineage_walks_base_ref(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        reg.register(CheckpointEntry("sft-1", "sft", "d1", "Qwen-base", created_at_ms=100))
        reg.register(CheckpointEntry("dpo-1", "dpo", "d2", "sft-1", created_at_ms=200))
        chain = reg.lineage("dpo-1")
        assert [c.checkpoint_id for c in chain] == ["dpo-1", "sft-1"]
        # base model is not a registered checkpoint → chain stops
        assert reg.lineage("sft-1")[0].base_ref == "Qwen-base"

    def test_lineage_unknown_is_empty(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        assert reg.lineage("nope") == []

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "checkpoints.jsonl"
        CheckpointRegistry(path).register(self._entry("sft-v1"))
        # new instance reading the same file sees the entry
        assert CheckpointRegistry(path).find("sft-v1") is not None

    def test_rejects_unknown_mode(self, tmp_path):
        reg = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
        with pytest.raises(ValueError, match="unknown mode"):
            reg.register(CheckpointEntry("x", "grpo", "d", "base"))


# ===========================================================================
# 3. Loss telemetry → trust
# ===========================================================================

class TestLossTelemetry:
    def test_parse_json_log_lines(self):
        lines = ['{"loss": 2.0, "step": 10}', "noise", '{"loss": 0.05, "step": 20}']
        t = parse_trainer_log(lines, mode="sft")
        assert t.loss_history == [2.0, 0.05]
        assert t.final_loss == 0.05
        assert t.initial_loss == 2.0
        assert t.converged is True

    def test_parse_keyvalue_log_line(self):
        t = parse_trainer_log(["loss = 0.05", "loss=0.2"], mode="sft")
        assert t.loss_history == [0.05, 0.2]

    def test_from_summary(self):
        t = telemetry_from_summary(
            {"final_loss": 0.05, "loss_history": [2.0, 0.05], "max_steps": 200}, mode="sft"
        )
        assert t.final_loss == 0.05
        assert t.steps == 200
        assert t.converged is True

    def test_health_zero_without_evidence(self):
        assert telemetry_to_training_health(LossTelemetry(mode="sft")) == 0.0
        assert telemetry_to_training_health(telemetry_from_loss_history([], mode="sft")) == 0.0

    def test_health_converged_run_is_high(self):
        t = telemetry_from_loss_history([2.0, 0.05], mode="sft")
        health = telemetry_to_training_health(t)
        assert 0.9 <= health <= 1.0
        assert t.converged is True

    def test_health_divergent_run_is_low(self):
        t = telemetry_from_loss_history([0.5, 1.5], mode="sft")
        health = telemetry_to_training_health(t)
        assert 0.0 <= health < 0.3
        assert t.converged is False

    def test_trust_builder_folds_telemetry(self):
        builder = AgentTrustScoreBuilder()
        builder.add_task_success(True)
        builder.add_training_telemetry(telemetry_from_loss_history([2.0, 0.05], mode="sft"))
        score = builder.build()
        assert "training_health" in score.breakdown
        assert score.breakdown["training_health"] is not None
        assert score.overall is not None
        assert score.overall > 0.0

    def test_trust_builder_accepts_float_and_ducktype(self):
        b1 = AgentTrustScoreBuilder().add_training_telemetry(0.9)
        assert b1.build().breakdown["training_health"] == pytest.approx(0.9)

        class _Duck:
            mode = "sft"
            steps = 2
            initial_loss = 2.0
            final_loss = 0.05
            loss_history = [2.0, 0.05]
            converged = True
            converged_below = 0.1

        b2 = AgentTrustScoreBuilder().add_training_telemetry(_Duck())
        assert b2.build().breakdown["training_health"] > 0.9


# ===========================================================================
# 4. Job manifest — wire exporters → orchestration
# ===========================================================================

class TestTrainingJobManifest:
    def test_build_writes_manifest_and_versioned_dataset(self, tmp_path):
        manifest, dv = build_training_job_manifest(
            records=_sft_dicts(2),
            mode="sft",
            out_dir=tmp_path,
            base_ref="Qwen/Qwen2.5-1.5B-Instruct",
            orchestrator="ray",
            hyperparameters={"lr": 1e-4},
            sources=["rollouts.jsonl"],
        )
        assert isinstance(manifest, TrainingJobManifest)
        assert manifest.mode == "sft"
        assert manifest.orchestrator == "ray"
        assert manifest.base_ref == "Qwen/Qwen2.5-1.5B-Instruct"
        assert manifest.dataset["content_digest"] == dv.content_digest
        assert manifest.checkpoint_plan["save_steps"] == 50
        assert manifest.hyperparameters == {"lr": 1e-4}
        assert manifest.telemetry_sink.endswith(".telemetry.json")
        # files written
        assert (tmp_path / "training-job.manifest.json").exists()
        assert (tmp_path / f"sft-{dv.version}.jsonl").exists()

    def test_records_for_mode_dispatches_exporters(self):
        branches = [
            _branch(branch_index=0, score=1, status="pass", answer="Good."),
            _branch(branch_index=1, score=0, status="fail", answer="Bad."),
        ]
        sft = records_for_mode(branches, "sft")
        dpo = records_for_mode(branches, "dpo")
        ppo = records_for_mode(branches, "ppo")
        # SFT: only the passing branch; DPO: one pair; PPO: both branches
        assert len(sft) == 1
        assert len(dpo) == 1
        assert len(ppo) == 2

    def test_records_for_mode_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown mode"):
            records_for_mode([], "grpo")

    def test_build_validates_mode_and_orchestrator(self, tmp_path):
        with pytest.raises(ValueError, match="unknown mode"):
            build_training_job_manifest(records=_sft_dicts(1), mode="grpo", out_dir=tmp_path)
        with pytest.raises(ValueError, match="unknown orchestrator"):
            build_training_job_manifest(
                records=_sft_dicts(1), mode="sft", out_dir=tmp_path, orchestrator="slurm"
            )

    def test_manifest_round_trip_json(self, tmp_path):
        manifest, _ = build_training_job_manifest(records=_sft_dicts(1), mode="ppo", out_dir=tmp_path)
        data = json.loads((tmp_path / "training-job.manifest.json").read_text())
        assert data["mode"] == "ppo"
        assert data["job_id"] == manifest.job_id

    def test_mode_and_orchestrator_constants(self):
        assert TRAINING_MODES == ("sft", "dpo", "ppo")
        assert "ray" in ORCHESTRATORS and "lightning" in ORCHESTRATORS and "local" in ORCHESTRATORS
