"""Tests for RouterRecord JSONL round-trip and export.py router integration."""
from __future__ import annotations

import json
import tempfile
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest

from evomerge.eval.metrics import EvalRecord
from evomerge.export import run_export
from evomerge.io import load_router_records, write_dicts_jsonl
from evomerge.router.features import RouterFeatures, feature_from_record
from evomerge.router.labels import (
    RouterLabel,
    RouterRecord,
)
from evomerge.schemas.training import Provenance
from evomerge.synthesize.templates import TaskType, make_task_spec


def _features() -> RouterFeatures:
    spec = make_task_spec(TaskType.tool_call, intent="test")
    return feature_from_record(spec)


def _record(task_id="t1", label=RouterLabel.small_model_can_handle) -> RouterRecord:
    return RouterRecord(
        task_id=task_id,
        features=_features(),
        label=label,
        provenance=Provenance(source="test", task_id=task_id),
    )


class TestRouterRecordRoundTrip:
    def test_to_dict_has_required_keys(self):
        r = _record()
        d = r.to_dict()
        assert "task_id" in d
        assert "features" in d
        assert "label" in d
        assert "provenance" in d

    def test_from_dict_restores_label(self):
        for label in RouterLabel:
            r = _record(label=label)
            d = r.to_dict()
            r2 = RouterRecord.from_dict(d)
            assert r2.label == label

    def test_from_dict_restores_features(self):
        r = _record()
        d = r.to_dict()
        r2 = RouterRecord.from_dict(d)
        assert r2.features.taskspec_n_constraints == r.features.taskspec_n_constraints
        assert r2.features.eval_tool_validity_rate == r.features.eval_tool_validity_rate

    def test_from_dict_restores_task_id(self):
        r = _record(task_id="custom-task")
        r2 = RouterRecord.from_dict(r.to_dict())
        assert r2.task_id == "custom-task"

    def test_all_feature_fields_survive_roundtrip(self):
        r = _record()
        d = r.to_dict()
        r2 = RouterRecord.from_dict(d)
        for f in dc_fields(RouterFeatures):
            assert getattr(r2.features, f.name) == getattr(r.features, f.name), f.name

    def test_jsonl_write_and_load(self):
        records = [_record(f"t{i}", list(RouterLabel)[i % 4]) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "router.jsonl"
            write_dicts_jsonl([r.to_dict() for r in records], p)
            loaded = load_router_records(p)
        assert len(loaded) == 8
        for orig, loaded_r in zip(records, loaded, strict=False):
            assert orig.task_id == loaded_r.task_id
            assert orig.label == loaded_r.label

    def test_jsonl_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "router.jsonl"
            with open(p, "w") as fh:
                fh.write(json.dumps(_record().to_dict()) + "\n")
                fh.write("\n")
                fh.write("# comment\n")
                fh.write(json.dumps(_record("t2").to_dict()) + "\n")
            loaded = load_router_records(p)
        assert len(loaded) == 2

    def test_invalid_label_raises(self):
        d = _record().to_dict()
        d["label"] = "not_a_real_label"
        with pytest.raises((ValueError, KeyError)):
            RouterRecord.from_dict(d)


class TestExportWithRouter:
    def _make_eval_records(self, task_ids: list[str]) -> list[EvalRecord]:
        return [
            EvalRecord(
                task_id=tid,
                group="C",
                final_pass=(i % 2 == 0),
                repair_rounds=i % 3,
            )
            for i, tid in enumerate(task_ids)
        ]

    def test_router_jsonl_written_when_specs_and_records_provided(self):
        task_ids = [f"t{i}" for i in range(6)]
        specs = {tid: make_task_spec(TaskType.repair, intent="test", task_id=tid) for tid in task_ids}
        eval_records = self._make_eval_records(task_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(
                out_dir=tmpdir,
                task_specs=specs,
                eval_records=eval_records,
            )
        assert "router" in manifest.files
        assert manifest.n_router == 6

    def test_router_jsonl_loadable(self):
        task_ids = [f"t{i}" for i in range(4)]
        specs = {tid: make_task_spec(TaskType.tool_call, intent="test", task_id=tid) for tid in task_ids}
        eval_records = self._make_eval_records(task_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(
                out_dir=tmpdir,
                task_specs=specs,
                eval_records=eval_records,
            )
            loaded = load_router_records(manifest.files["router"])
        assert len(loaded) == 4
        assert all(isinstance(r.label, RouterLabel) for r in loaded)

    def test_no_router_without_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(out_dir=tmpdir)
        assert "router" not in manifest.files
        assert manifest.n_router == 0

    def test_no_router_without_eval_records(self):
        task_ids = ["t1"]
        specs = {"t1": make_task_spec(TaskType.repair, intent="test")}
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(out_dir=tmpdir, task_specs=specs)
        assert "router" not in manifest.files

    def test_manifest_has_n_router_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(out_dir=tmpdir)
        d = manifest.to_dict()
        assert "n_router" in d

    def test_manifest_json_has_n_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_export(out_dir=tmpdir)
            mf_path = Path(manifest.files["manifest"])
            d = json.loads(mf_path.read_text())
        assert "n_router" in d
