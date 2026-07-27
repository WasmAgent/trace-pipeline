"""Tests for the evomerge CLI (__main__.py)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evomerge.__main__ import main
from evomerge.io import write_dicts_jsonl
from evomerge.router.features import feature_from_record
from evomerge.router.labels import RouterLabel, RouterRecord
from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import Message, Provenance, SftTrainingRecord
from evomerge.synthesize.templates import TaskType, make_task_spec


def _rollout_fixture(tmpdir: str, n: int = 4) -> str:
    """Write n rollout records to a JSONL file, alternating pass/fail."""
    p = Path(tmpdir) / "rollouts.jsonl"
    with open(p, "w") as fh:
        for i in range(n):
            rec = RolloutBranchRecord(
                rollout_id=f"r{i // 2}",
                task=f"task {i // 2}",
                branch_index=i % 2,
                temperature=0.7,
                session_id="s1",
                final_answer=f"answer {i}",
                objective_score=i % 2,
                objective_status="pass" if i % 2 else "fail",
                rank=i % 2,
                total_score=float(i % 2),
            )
            fh.write(rec.model_dump_json() + "\n")
    return str(p)


def _router_fixture(tmpdir: str) -> str:
    """Write router records for testing."""
    spec = make_task_spec(TaskType.repair, intent="test")
    records = [
        RouterRecord(
            task_id=f"t{i}",
            features=feature_from_record(spec),
            label=list(RouterLabel)[i % 4],
            provenance=Provenance(source="test", task_id=f"t{i}"),
        )
        for i in range(6)
    ]
    p = Path(tmpdir) / "router.jsonl"
    write_dicts_jsonl([r.to_dict() for r in records], p)
    return str(p)


def _sft_fixture(tmpdir: str) -> str:
    """Write SFT records."""
    p = Path(tmpdir) / "sft.jsonl"
    prov = Provenance(source="test")
    records = [
        SftTrainingRecord(
            messages=[
                Message(role="user", content=f"task {i}"),
                Message(role="assistant", content=f"answer {i}"),
            ],
            output_type="final_answer",
            provenance=prov,
        )
        for i in range(3)
    ]
    from evomerge.io import write_jsonl
    write_jsonl(records, p)
    return str(p)


class TestExportCommand:
    def test_export_with_rollout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_file = _rollout_fixture(tmpdir)
            out_dir = str(Path(tmpdir) / "out")
            rc = main(["export", "--rollout", rollout_file, "--out-dir", out_dir])
        assert rc == 0

    def test_export_creates_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_file = _rollout_fixture(tmpdir)
            out_dir = str(Path(tmpdir) / "out")
            main(["export", "--rollout", rollout_file, "--out-dir", out_dir])
            manifest_path = Path(out_dir) / "manifest.json"
            assert manifest_path.exists()
            d = json.loads(manifest_path.read_text())
            assert "n_sft" in d
            assert "n_dpo" in d
            assert "n_router" in d

    def test_export_no_inputs_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "out")
            rc = main(["export", "--out-dir", out_dir])
        assert rc == 0

    def test_export_missing_eval_items_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "out")
            rc = main(["export", "--out-dir", out_dir, "--eval-items", "/nonexistent.jsonl"])
        assert rc == 1

    def test_export_include_failing_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_file = _rollout_fixture(tmpdir)
            out_dir = str(Path(tmpdir) / "out")
            rc = main([
                "export", "--rollout", rollout_file,
                "--out-dir", out_dir, "--include-failing",
            ])
        assert rc == 0


class TestRouterCommand:
    def test_router_predict_stdout(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            router_file = _router_fixture(tmpdir)
            rc = main(["router", "--input", router_file])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        assert d["n"] == 6
        assert "accuracy" in d
        assert len(d["predictions"]) == 6

    def test_router_predict_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router_file = _router_fixture(tmpdir)
            out_file = str(Path(tmpdir) / "predictions.json")
            rc = main(["router", "--input", router_file, "--out", out_file])
            assert rc == 0
            assert Path(out_file).exists()
            d = json.loads(Path(out_file).read_text())
        assert d["n"] == 6

    def test_router_predictions_have_correct_keys(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            router_file = _router_fixture(tmpdir)
            main(["router", "--input", router_file])
        out = capsys.readouterr().out
        predictions = json.loads(out)["predictions"]
        for p in predictions:
            assert "task_id" in p
            assert "predicted_label" in p
            assert "stored_label" in p
            assert "reason" in p
            assert "correct" in p

    def test_router_custom_thresholds(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            router_file = _router_fixture(tmpdir)
            rc = main([
                "router", "--input", router_file,
                "--max-repair-rounds", "1",
                "--min-tool-validity", "0.95",
            ])
        assert rc == 0


class TestValidateCommand:
    def test_validate_valid_sft(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_file = _sft_fixture(tmpdir)
            rc = main(["validate", "--input", sft_file])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        assert d["n_invalid"] == 0

    def test_validate_strict_with_valid_data_exits_0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_file = _sft_fixture(tmpdir)
            rc = main(["validate", "--input", sft_file, "--strict"])
        assert rc == 0

    def test_validate_router_jsonl_skips_schema_check(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            router_file = _router_fixture(tmpdir)
            rc = main(["validate", "--input", router_file])
        assert rc == 0

    def test_validate_reports_n_records(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_file = _sft_fixture(tmpdir)
            main(["validate", "--input", sft_file])
        out = capsys.readouterr().out
        d = json.loads(out)
        assert d["n_records"] == 3


class TestCLIHelp:
    def test_no_command_exits_0(self):
        rc = main([])
        assert rc == 0

    def test_export_help_exits_0(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--help"])
        assert exc_info.value.code == 0

    def test_router_help_exits_0(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["router", "--help"])
        assert exc_info.value.code == 0

    def test_validate_help_exits_0(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "--help"])
        assert exc_info.value.code == 0
