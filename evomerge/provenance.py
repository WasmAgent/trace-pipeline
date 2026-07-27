"""Run provenance — lightweight receipt for pipeline runs.

Inspired by SCITT/in-toto but without full PKI. Each export run produces
a RunReceipt JSON file recording:
  - run_id, timestamp, operator
  - repo_commit (git HEAD)
  - input file digests (SHA-256)
  - output file digests (SHA-256)
  - policy bundle digest (if provided)
  - model_id(s) used
  - evomerge version

Usage:
    from evomerge.provenance import RunReceiptBuilder, compute_file_digest

    builder = RunReceiptBuilder(run_id="export-2026-06-25-001", operator="ci")
    builder.add_input("data/rollouts.jsonl")
    builder.add_output("data/training/sft.jsonl")
    receipt = builder.build()
    receipt.save(Path("data/training/run-receipt.json"))
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


def compute_file_digest(path: Path) -> str:
    """SHA-256 hex digest of a file. Returns empty string if file not found."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str:
    """Return the current git HEAD commit SHA, or empty string if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _evomerge_version() -> str:
    try:
        from evomerge import __version__  # type: ignore[attr-defined]
        return __version__
    except (ImportError, AttributeError):
        return "unknown"


@dataclass
class FileRef:
    path: str
    digest: str  # sha256 hex


@dataclass
class RunReceipt:
    """Immutable record of one pipeline run."""
    receipt_version: str
    run_id: str
    timestamp_utc: str
    operator: str
    repo_commit: str
    evomerge_version: str
    inputs: list[FileRef] = field(default_factory=list)
    outputs: list[FileRef] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    policy_bundle_digest: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunReceipt:
        data = json.loads(path.read_text())
        inputs = [FileRef(**f) for f in data.pop("inputs", [])]
        outputs = [FileRef(**f) for f in data.pop("outputs", [])]
        return cls(**data, inputs=inputs, outputs=outputs)

    @property
    def receipt_digest(self) -> str:
        """SHA-256 of the canonical JSON form of this receipt."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class RunReceiptBuilder:
    """Build a RunReceipt incrementally."""

    def __init__(self, run_id: str, operator: str = "unknown", notes: str = "") -> None:
        import time
        self._run_id = run_id
        self._operator = operator
        self._notes = notes
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._inputs: list[FileRef] = []
        self._outputs: list[FileRef] = []
        self._model_ids: list[str] = []
        self._policy_bundle_digest = ""

    def add_input(self, path: str | Path) -> RunReceiptBuilder:
        p = Path(path)
        self._inputs.append(FileRef(path=str(p), digest=compute_file_digest(p)))
        return self

    def add_output(self, path: str | Path) -> RunReceiptBuilder:
        p = Path(path)
        self._outputs.append(FileRef(path=str(p), digest=compute_file_digest(p)))
        return self

    def add_model(self, model_id: str) -> RunReceiptBuilder:
        if model_id not in self._model_ids:
            self._model_ids.append(model_id)
        return self

    def set_policy_bundle(self, digest: str) -> RunReceiptBuilder:
        self._policy_bundle_digest = digest
        return self

    def build(self) -> RunReceipt:
        return RunReceipt(
            receipt_version="run-receipt/v0.1",
            run_id=self._run_id,
            timestamp_utc=self._timestamp,
            operator=self._operator,
            repo_commit=_git_head_sha(),
            evomerge_version=_evomerge_version(),
            inputs=self._inputs,
            outputs=self._outputs,
            model_ids=self._model_ids,
            policy_bundle_digest=self._policy_bundle_digest,
            notes=self._notes,
        )
