"""Integration smoke test: full data-loop from fixture JSONL to training output.

Uses fixtures/data-loop/rollout-branches.v1.jsonl — the same file that is
byte-identical across evomerge-framework, wasmagent-js, and bscode.

Expected counts (from fixtures/data-loop/manifest.json):
  n_sft  = 1  (only the passing branch, branch_index=0)
  n_dpo  = 1  (one chosen/rejected pair)
  n_ppo  = 2  (both branches get a reward record)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "data-loop"
FIXTURE_JSONL = FIXTURE_DIR / "rollout-branches.v1.jsonl"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.json"


@pytest.fixture(scope="module")
def fixture_manifest() -> dict:
    return json.loads(FIXTURE_MANIFEST.read_text())


class TestFixtureIntegrity:
    def test_fixture_file_exists(self):
        assert FIXTURE_JSONL.exists(), f"fixture not found: {FIXTURE_JSONL}"

    def test_fixture_has_two_branches(self):
        lines = [l for l in FIXTURE_JSONL.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_fixture_schema_version(self):
        from evomerge.io import load_rollouts
        branches = load_rollouts(FIXTURE_JSONL)
        for b in branches:
            assert b.schema_version == "rollout-wire/v1"

    def test_fixture_one_pass_one_fail(self):
        from evomerge.io import load_rollouts
        branches = load_rollouts(FIXTURE_JSONL)
        scores = {b.objective_score for b in branches}
        assert scores == {0, 1}

    def test_fixture_same_rollout_id(self):
        from evomerge.io import load_rollouts
        branches = load_rollouts(FIXTURE_JSONL)
        ids = {b.rollout_id for b in branches}
        assert len(ids) == 1


class TestDataLoopPipeline:
    def test_sft_only_passing_branch(self, fixture_manifest):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.sft import to_sft_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_sft_records(branches)
        assert len(records) == fixture_manifest["expected"]["n_sft"]

    def test_dpo_one_pair(self, fixture_manifest):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.dpo import to_dpo_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_dpo_records(branches)
        assert len(records) == fixture_manifest["expected"]["n_dpo"]

    def test_dpo_chosen_is_passing_branch(self, fixture_manifest):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.dpo import to_dpo_records
        branches = load_rollouts(FIXTURE_JSONL)
        passing = next(b for b in branches if b.objective_score == 1)
        failing = next(b for b in branches if b.objective_score == 0)
        records = to_dpo_records(branches)
        assert records[0].chosen == passing.final_answer
        assert records[0].rejected == failing.final_answer

    def test_ppo_both_branches(self, fixture_manifest):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.ppo import to_ppo_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_ppo_records(branches)
        assert len(records) == fixture_manifest["expected"]["n_ppo"]

    def test_ppo_rewards_match_scores(self):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.ppo import to_ppo_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_ppo_records(branches)
        score_map = {b.branch_index: b.objective_score for b in branches}
        for rec in records:
            # provenance links back via rollout_id; reward == float(objective_score)
            assert rec.reward in (0.0, 1.0)
        rewards = sorted(r.reward for r in records)
        assert rewards == [0.0, 1.0]

    def test_sft_message_structure(self):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.sft import to_sft_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_sft_records(branches)
        rec = records[0]
        assert rec.messages[0].role == "user"
        assert rec.messages[-1].role == "assistant"
        assert rec.messages[-1].content != ""

    def test_sft_provenance(self):
        from evomerge.io import load_rollouts
        from evomerge.pipeline.sft import to_sft_records
        branches = load_rollouts(FIXTURE_JSONL)
        records = to_sft_records(branches)
        assert records[0].provenance.source == "wasmagent-rollout"
        assert records[0].provenance.rollout_id is not None


class TestRunExportIntegration:
    def test_full_export_produces_expected_files(self, fixture_manifest):
        from evomerge.export import run_export
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(
                rollout_jsonl=str(FIXTURE_JSONL),
                out_dir=tmpdir,
            )
            assert manifest.n_sft == fixture_manifest["expected"]["n_sft"]
            assert manifest.n_dpo == fixture_manifest["expected"]["n_dpo"]
            assert manifest.n_ppo == fixture_manifest["expected"]["n_ppo"]
            assert manifest.n_invalid == 0
            assert Path(manifest.files["manifest"]).exists()

    def test_exported_sft_is_valid_jsonl(self):
        from evomerge.export import run_export
        from evomerge.schemas.training import SftTrainingRecord
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(rollout_jsonl=str(FIXTURE_JSONL), out_dir=tmpdir)
            sft_path = Path(manifest.files["sft"])
            with open(sft_path) as fh:
                for line in fh:
                    rec = SftTrainingRecord.model_validate_json(line.strip())
                    assert rec.schema_version == "sft/v1"

    def test_exported_dpo_chosen_ne_rejected(self):
        from evomerge.export import run_export
        from evomerge.schemas.training import DpoTrainingRecord
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(rollout_jsonl=str(FIXTURE_JSONL), out_dir=tmpdir)
            dpo_path = Path(manifest.files["dpo"])
            with open(dpo_path) as fh:
                for line in fh:
                    rec = DpoTrainingRecord.model_validate_json(line.strip())
                    assert rec.chosen != rec.rejected

    def test_contamination_check_no_false_positives(self):
        from evomerge.export import run_export
        eval_texts = ["An unrelated sentence about dogs and cats."]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(
                rollout_jsonl=str(FIXTURE_JSONL),
                out_dir=tmpdir,
                eval_texts=eval_texts,
            )
        assert manifest.n_contaminated == 0

    def test_manifest_json_matches_manifest_object(self):
        from evomerge.export import run_export
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(rollout_jsonl=str(FIXTURE_JSONL), out_dir=tmpdir)
            mf_path = Path(manifest.files["manifest"])
            d = json.loads(mf_path.read_text())
        assert d["n_sft"] == manifest.n_sft
        assert d["n_dpo"] == manifest.n_dpo
        assert d["n_ppo"] == manifest.n_ppo


class TestCLIIntegration:
    def test_cli_export_with_fixture(self):
        from evomerge.__main__ import main
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = main([
                "export",
                "--rollout", str(FIXTURE_JSONL),
                "--out-dir", tmpdir,
            ])
        assert rc == 0

    def test_cli_validate_exported_sft(self, capsys):
        from evomerge.export import run_export
        from evomerge.__main__ import main
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(rollout_jsonl=str(FIXTURE_JSONL), out_dir=tmpdir)
            rc = main(["validate", "--input", manifest.files["sft"], "--strict"])
            out = capsys.readouterr().out
        assert rc == 0
        d = json.loads(out)
        assert d["n_invalid"] == 0
