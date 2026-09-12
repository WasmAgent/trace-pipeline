"""Consumer-side corpus gate: trace-pipeline executes the central manifest.

When AEP_PROTOCOL_CORPUS points at the wasmagent-protocol conformance
checkout, this test enforces the manifest's structural and semantic
expectations with trace-pipeline's own validator — the shared corpus is a
live consumer-side gate, not protocol-side data. Without the variable the
module is skipped (e.g. hermetic local runs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

CORPUS = os.environ.get("AEP_PROTOCOL_CORPUS")

pytestmark = pytest.mark.skipif(
    not CORPUS or not Path(CORPUS, "manifest.json").exists(),
    reason="AEP_PROTOCOL_CORPUS not set (central corpus not checked out)",
)


def _records(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    return [json.loads(path.read_text(encoding="utf-8"))]


def test_central_corpus_structural_and_semantic() -> None:
    from evomerge.validate.aep import validate_aep_record

    manifest = json.loads((Path(CORPUS) / "manifest.json").read_text(encoding="utf8"))
    entries = manifest.get("conformance_target", [])
    assert entries, "manifest declares no current-target fixtures"

    executed = 0
    for entry in entries:
        fixture = Path(CORPUS) / entry["path"]
        structural = entry.get("structural")
        semantic = entry.get("semantic")
        if structural not in ("valid", "invalid") and semantic not in ("valid", "invalid"):
            continue

        for record in _records(fixture):
            result = validate_aep_record(record)
            if structural in ("valid", "invalid"):
                schema_ok = result.valid_schema
                assert schema_ok is (structural == "valid"), (
                    f"{entry['path']}: structural expected {structural}, "
                    f"errors={result.errors[:3]}"
                )
                executed += 1
            # Semantic expectations are only reachable on structurally valid
            # records — the schema enum/minimum reject unknown grades and
            # negative counts before the semantic layer runs.
            if semantic in ("valid", "invalid") and result.valid_schema:
                assert result.semantic_valid is (semantic == "valid"), (
                    f"{entry['path']}: semantic expected {semantic}, errors={result.errors[:3]}"
                )
                executed += 1

    assert executed >= 20, f"corpus executed too few checks: {executed}"
