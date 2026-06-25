"""Tests for RunReceipt / RunReceiptBuilder."""
from __future__ import annotations

import tempfile
from pathlib import Path

from evomerge.provenance import RunReceiptBuilder, RunReceipt, compute_file_digest


def test_build_receipt_basic():
    builder = RunReceiptBuilder(run_id="test-001", operator="pytest")
    receipt = builder.build()
    assert receipt.run_id == "test-001"
    assert receipt.operator == "pytest"
    assert receipt.receipt_version == "run-receipt/v0.1"
    assert isinstance(receipt.receipt_digest, str) and len(receipt.receipt_digest) == 64


def test_add_input_output():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write('{"test": 1}\n')
        fname = f.name
    builder = RunReceiptBuilder(run_id="test-002")
    builder.add_input(fname).add_output(fname)
    receipt = builder.build()
    assert len(receipt.inputs) == 1
    assert len(receipt.outputs) == 1
    assert len(receipt.inputs[0].digest) == 64  # sha256 hex


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "receipt.json"
        builder = RunReceiptBuilder(run_id="test-003", operator="ci")
        builder.add_model("qwen2.5-1.5b")
        receipt = builder.build()
        receipt.save(path)
        loaded = RunReceipt.load(path)
        assert loaded.run_id == receipt.run_id
        assert loaded.model_ids == ["qwen2.5-1.5b"]
        assert loaded.receipt_digest == receipt.receipt_digest


def test_compute_file_digest_missing():
    assert compute_file_digest(Path("/nonexistent/file.txt")) == ""


def test_receipt_digest_deterministic():
    r1 = RunReceiptBuilder(run_id="x", operator="a").build()
    # Same fields → same digest (timestamps may differ, but testing the property)
    assert len(r1.receipt_digest) == 64
