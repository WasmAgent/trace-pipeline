"""evomerge.io — JSONL read / write helpers for all record types."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: str | Path, model: Type[T]) -> list[T]:
    """Parse a JSONL file into validated Pydantic records.

    Skips blank lines and comment lines starting with '#'.
    """
    records: list[T] = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def write_jsonl(
    records: Sequence[BaseModel],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Serialise Pydantic records to a JSONL file.

    Returns:
        Number of records written.
    """
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    return len(records)


def write_dicts_jsonl(
    records: Sequence[dict],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Serialise plain dicts (e.g. RouterRecord.to_dict()) to JSONL.

    Returns:
        Number of records written.
    """
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def load_router_records(path: str | Path):
    """Load RouterRecord list from JSONL."""
    from evomerge.router.labels import RouterRecord
    records = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(RouterRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def load_rollouts(path: str | Path):
    """Convenience: load RolloutBranchRecord list from JSONL.

    Gracefully skips RolloutTreeRecord entries (which carry fork topology
    but are not branch-level records).
    """
    import json as _json
    import logging

    _log = logging.getLogger(__name__)
    from evomerge.schemas.rollout import RolloutBranchRecord

    records: list[RolloutBranchRecord] = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = _json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            # Skip tree records (they have fork_map but no branch_index)
            if "fork_map" in obj and "branch_index" not in obj:
                _log.info(
                    "load_rollouts: skipping RolloutTreeRecord at %s:%d", path, lineno
                )
                continue
            try:
                records.append(RolloutBranchRecord.model_validate(obj))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def load_compliance_records(path: str | Path):
    """Convenience: load ComplianceEvalRecord list from JSONL."""
    from evomerge.schemas.compliance import ComplianceEvalRecord
    return load_jsonl(path, ComplianceEvalRecord)


__all__ = [
    "load_compliance_records",
    "load_jsonl",
    "load_rollouts",
    "load_router_records",
    "write_dicts_jsonl",
    "write_jsonl",
]
