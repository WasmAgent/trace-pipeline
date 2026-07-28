"""Training-pipeline integration — wire trace exporters to training-job orchestration.

This module is the integration layer between the validated trace exporters
(``evomerge.pipeline.to_sft_records`` / ``to_dpo_records`` / ``to_ppo_records``)
and an external training-job orchestrator (Ray / Lightning / the local TRL shims
in ``scripts/train_*.py``).

Per the repo boundary (CLAUDE.md), trace-pipeline produces training *data* and
thin orchestration *contracts* — it is **not** a training framework.  This
module therefore provides four data-side primitives that together "wire the
validated trace exporters to actual training job orchestration":

  1. Dataset versioning    — content-addressed dataset manifests (``version_dataset``)
  2. Checkpoint management — append-only lineage registry (``CheckpointRegistry``)
  3. Loss telemetry        — training-loss → trust-score feedback (``LossTelemetry``)
  4. Job manifest          — wires exporters → versioned dataset → checkpoint plan
                             → telemetry sink → orchestrator (``build_training_job_manifest``)

The actual Ray/Lightning cluster execution lives outside this repo; the
``TrainingJobManifest`` is the contract an external runner consumes.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evomerge.io import write_dicts_jsonl

TRAINING_MODES: tuple[str, ...] = ("sft", "dpo", "ppo")
ORCHESTRATORS: tuple[str, ...] = ("local", "ray", "lightning")

#: Default convergence threshold for ``LossTelemetry`` (mirrors train_sft.py).
_DEFAULT_CONVERGED_BELOW = 0.1


# ===========================================================================
# Helpers
# ===========================================================================

def _record_to_dict(rec: Any) -> dict:
    """Normalise a training record (pydantic model or plain dict) to a dict."""
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "model_dump"):  # pydantic v2
        return rec.model_dump(mode="json")
    if hasattr(rec, "dict"):  # pydantic v1 fallback
        return rec.dict()
    return dict(rec)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_version(digest: str) -> str:
    return "v-" + digest[:6]


def _now_ms() -> int:
    return int(time.time() * 1000)


# ===========================================================================
# 1. Dataset versioning
# ===========================================================================

def compute_dataset_digest(records: Iterable[Any]) -> str:
    """SHA-256 over the canonical serialisation of every record.

    Deterministic: identical record content yields an identical digest
    regardless of dict key ordering, so re-versioning unchanged data produces
    the same version tag (idempotent).
    """
    h = hashlib.sha256()
    for rec in records:
        h.update(_canonical_json(_record_to_dict(rec)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class DatasetVersion:
    """Content-addressed version of a training-record dataset."""

    name: str                     # 'sft' | 'dpo' | 'ppo'
    schema_version: str           # e.g. 'sft/v1'
    content_digest: str           # full sha256
    version: str                  # short tag, e.g. 'v-a1b2c3'
    record_count: int
    path: str                     # versioned jsonl path
    manifest_path: str            # sidecar manifest json path
    created_at_ms: int
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def version_dataset(
    records: Sequence[Any],
    *,
    name: str,
    out_dir: str | Path,
    sources: list[str] | None = None,
) -> DatasetVersion:
    """Write a content-addressed, versioned copy of ``records`` plus a manifest.

    The output filename embeds the short content digest, so the same record
    content always maps to the same versioned path (idempotent).  A sidecar
    ``<name>-<version>.manifest.json`` records the schema version, record count,
    full digest, source lineage, and creation time.

    Args:
        records: training records (pydantic models or plain dicts).
        name: dataset name — one of ``TRAINING_MODES``.
        out_dir: directory to write into (created if absent).
        sources: optional lineage strings (e.g. source rollout JSONL paths).
    """
    if name not in TRAINING_MODES:
        raise ValueError(f"unknown training mode {name!r}; expected one of {TRAINING_MODES}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dicts = [_record_to_dict(r) for r in records]
    digest = compute_dataset_digest(dicts)
    version = _short_version(digest)
    schema_version = (dicts[0].get("schema_version") if dicts else None) or f"{name}/v1"

    data_path = out / f"{name}-{version}.jsonl"
    manifest_path = out / f"{name}-{version}.manifest.json"

    write_dicts_jsonl(dicts, data_path)
    dv = DatasetVersion(
        name=name,
        schema_version=schema_version,
        content_digest=digest,
        version=version,
        record_count=len(dicts),
        path=str(data_path),
        manifest_path=str(manifest_path),
        created_at_ms=_now_ms(),
        sources=list(sources or []),
    )
    manifest_path.write_text(json.dumps(dv.to_dict(), indent=2, ensure_ascii=False))
    return dv


def load_dataset_manifest(path: str | Path) -> DatasetVersion:
    """Load a ``DatasetVersion`` previously written by :func:`version_dataset`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetVersion(**data)


# ===========================================================================
# 2. Checkpoint management
# ===========================================================================

@dataclass
class CheckpointEntry:
    """One checkpoint in the lineage registry."""

    checkpoint_id: str           # stable id, e.g. '<mode>-<dataset_version>'
    mode: str                    # 'sft' | 'dpo' | 'ppo'
    dataset_version: str         # content digest of the training dataset
    base_ref: str                # base model name OR parent checkpoint_id
    status: str = "ready"        # 'training' | 'ready' | 'failed'
    path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CheckpointRegistry:
    """Append-only JSONL registry of training checkpoints with lineage lookup.

    The registry is a simple append-only log; each ``checkpoint_id`` is recorded
    at most once.  Lineage is reconstructed by walking ``base_ref`` pointers back
    from a checkpoint to its base model.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _read(self) -> list[CheckpointEntry]:
        entries: list[CheckpointEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(CheckpointEntry(**json.loads(line)))
        return entries

    def all(self) -> list[CheckpointEntry]:
        return self._read()

    def find(self, checkpoint_id: str) -> CheckpointEntry | None:
        for entry in self._read():
            if entry.checkpoint_id == checkpoint_id:
                return entry
        return None

    def latest(self, mode: str | None = None) -> CheckpointEntry | None:
        entries = [e for e in self._read() if mode is None or e.mode == mode]
        if not entries:
            return None
        return max(entries, key=lambda e: (e.created_at_ms, e.checkpoint_id))

    def register(self, entry: CheckpointEntry) -> CheckpointEntry:
        """Append ``entry`` unless its checkpoint_id is already registered."""
        if entry.mode not in TRAINING_MODES:
            raise ValueError(f"unknown mode {entry.mode!r}; expected one of {TRAINING_MODES}")
        if not entry.created_at_ms:
            entry.created_at_ms = _now_ms()
        if self.find(entry.checkpoint_id) is None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def lineage(self, checkpoint_id: str) -> list[CheckpointEntry]:
        """Return the chain ``[ckpt, parent, ...]`` following ``base_ref`` pointers.

        Stops when ``base_ref`` no longer names a registered checkpoint (i.e. it
        is a base model name).  Cycle-safe.
        """
        by_id = {e.checkpoint_id: e for e in self._read()}
        chain: list[CheckpointEntry] = []
        seen: set[str] = set()
        cur = by_id.get(checkpoint_id)
        while cur is not None and cur.checkpoint_id not in seen:
            seen.add(cur.checkpoint_id)
            chain.append(cur)
            cur = by_id.get(cur.base_ref)
        return chain


# ===========================================================================
# 3. Training-loss telemetry → trust
# ===========================================================================

@dataclass
class LossTelemetry:
    """Training-loss telemetry extracted from a training run."""

    mode: str
    steps: int = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    loss_history: list[float] = field(default_factory=list)
    converged: bool = False
    converged_below: float = _DEFAULT_CONVERGED_BELOW

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_losses_from_log(lines: Iterable[str]) -> list[float]:
    """Pull ``loss`` values from HF Trainer-style log lines.

    Handles two shapes:
      - JSON-object lines: ``{"loss": 0.42, "step": 10}``
      - key=value lines:   ``'loss' = 0.42``  /  ``loss=0.42``  /  ``loss: 0.42``
    """
    losses: list[float] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        loss: float | None = None
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and obj.get("loss") is not None:
                    loss = float(obj["loss"])
            except (json.JSONDecodeError, TypeError, ValueError):
                loss = None
        if loss is None:
            low = s.lower()
            idx = low.find("loss")
            if idx != -1:
                rest = s[idx + 4:].lstrip("_=: \t'\"")
                num = ""
                for ch in rest:
                    if ch in "-.0123456789eE":
                        num += ch
                    else:
                        break
                if num and num not in ("-", "."):
                    try:
                        loss = float(num)
                    except ValueError:
                        loss = None
        if loss is not None:
            losses.append(loss)
    return losses


def telemetry_from_loss_history(
    loss_history: Sequence[float],
    *,
    mode: str,
    steps: int | None = None,
    converged_below: float = _DEFAULT_CONVERGED_BELOW,
) -> LossTelemetry:
    """Build telemetry from an explicit loss history."""
    hist = [float(x) for x in loss_history]
    final = hist[-1] if hist else None
    initial = hist[0] if hist else None
    return LossTelemetry(
        mode=mode,
        steps=steps if steps is not None else len(hist),
        initial_loss=initial,
        final_loss=final,
        loss_history=hist,
        converged=bool(hist) and final is not None and final < converged_below,
        converged_below=converged_below,
    )


def parse_trainer_log(
    lines: Iterable[str],
    *,
    mode: str,
    converged_below: float = _DEFAULT_CONVERGED_BELOW,
) -> LossTelemetry:
    """Build telemetry from HF Trainer log lines (see ``_extract_losses_from_log``)."""
    return telemetry_from_loss_history(
        _extract_losses_from_log(lines), mode=mode, converged_below=converged_below
    )


def telemetry_from_summary(
    summary: dict,
    *,
    mode: str,
    converged_below: float = _DEFAULT_CONVERGED_BELOW,
) -> LossTelemetry:
    """Build telemetry from a ``training_summary.json``-style dict.

    Recognises ``final_loss`` / ``loss`` / ``train_loss`` and optional
    ``loss_history`` / ``initial_loss`` / ``steps`` / ``max_steps``.
    """
    hist = [float(x) for x in (summary.get("loss_history") or [])]
    final = None
    for key in ("final_loss", "loss", "train_loss"):
        v = summary.get(key)
        if v is not None:
            try:
                final = float(v)
            except (TypeError, ValueError):
                pass
            break
    initial = summary.get("initial_loss")
    if initial is not None:
        try:
            initial = float(initial)
        except (TypeError, ValueError):
            initial = None
    if not hist:
        if final is not None and initial is not None:
            hist = [initial, final]
        elif final is not None:
            hist = [final]
    steps_raw = summary.get("steps") or summary.get("max_steps")
    try:
        steps = int(steps_raw) if steps_raw is not None else (len(hist) or 0)
    except (TypeError, ValueError):
        steps = len(hist) or 0
    return LossTelemetry(
        mode=mode,
        steps=steps,
        initial_loss=initial,
        final_loss=final,
        loss_history=hist,
        converged=final is not None and final < converged_below,
        converged_below=converged_below,
    )


def telemetry_to_training_health(telemetry: LossTelemetry) -> float:
    """Map a :class:`LossTelemetry` to a ``[0, 1]`` training-health score.

    Combines three signals:
      - convergence  (final_loss at/below the threshold → 1.0; decays to 0 at 2x)
      - improvement  (relative drop from initial → final; 0 if it got worse)
      - stability    (low relative variance in the tail of the history)

    Returns 0.0 when there is no loss evidence at all.
    """
    if not telemetry.loss_history or telemetry.final_loss is None:
        return 0.0

    final = telemetry.final_loss
    thr = telemetry.converged_below if telemetry.converged_below > 0 else _DEFAULT_CONVERGED_BELOW
    if final <= thr:
        convergence = 1.0
    elif final >= 2 * thr:
        convergence = 0.0
    else:
        convergence = (2 * thr - final) / thr

    if telemetry.initial_loss and telemetry.initial_loss > 0:
        improvement = max(0.0, min(1.0, (telemetry.initial_loss - final) / telemetry.initial_loss))
    else:
        improvement = 0.5  # no initial → neutral

    tail = telemetry.loss_history[max(1, len(telemetry.loss_history) // 2):]
    if len(tail) >= 2 and final > 0:
        mean = sum(tail) / len(tail)
        var = sum((x - mean) ** 2 for x in tail) / len(tail)
        stability = max(0.0, 1.0 - (var ** 0.5) / max(mean, 1e-9))
    else:
        stability = 0.8

    return max(0.0, min(1.0, convergence * 0.5 + improvement * 0.3 + stability * 0.2))


# ===========================================================================
# 4. Job manifest — wire exporters → orchestration
# ===========================================================================

@dataclass
class TrainingJobManifest:
    """Contract consumed by an external training orchestrator (Ray/Lightning/local).

    trace-pipeline produces the manifest; the actual cluster execution is out of
    scope (CLAUDE.md: not a training framework).  The manifest pins the versioned
    dataset, checkpoint plan, and telemetry sink so a run is reproducible and its
    loss telemetry can flow back into trust scores.
    """

    job_id: str
    mode: str
    orchestrator: str
    dataset: dict            # DatasetVersion.to_dict()
    base_ref: str
    checkpoint_plan: dict    # {save_steps, save_total_limit, resume_from}
    telemetry_sink: str      # path where LossTelemetry json is written
    hyperparameters: dict
    created_at_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


def records_for_mode(rollouts: Sequence[Any], mode: str) -> list[Any]:
    """Dispatch to the validated trace exporter for ``mode``.

    This is the explicit wire from the ``evomerge.pipeline`` converters into the
    orchestration manifest.  ``mode='dpo'`` preserves the converter's default
    attestation gate (branches without verifier_results are implicitly trusted).
    """
    if mode == "sft":
        from evomerge.pipeline import to_sft_records
        return to_sft_records(rollouts)
    if mode == "dpo":
        from evomerge.pipeline import to_dpo_records
        return to_dpo_records(rollouts)
    if mode == "ppo":
        from evomerge.pipeline import to_ppo_records
        return to_ppo_records(rollouts)
    raise ValueError(f"unknown mode {mode!r}; expected one of {TRAINING_MODES}")


def build_training_job_manifest(
    *,
    records: Sequence[Any],
    mode: str,
    out_dir: str | Path,
    base_ref: str = "",
    orchestrator: str = "local",
    hyperparameters: dict | None = None,
    save_steps: int = 50,
    save_total_limit: int = 3,
    resume_from: str | None = None,
    sources: list[str] | None = None,
) -> tuple[TrainingJobManifest, DatasetVersion]:
    """Version a dataset and emit a training-job manifest for an orchestrator.

    Writes:
      - ``<out>/<mode>-<version>.jsonl``           (versioned dataset)
      - ``<out>/<mode>-<version>.manifest.json``   (dataset manifest)
      - ``<out>/training-job.manifest.json``       (this job manifest)

    Returns ``(manifest, dataset_version)``.
    """
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {TRAINING_MODES}")
    if orchestrator not in ORCHESTRATORS:
        raise ValueError(
            f"unknown orchestrator {orchestrator!r}; expected one of {ORCHESTRATORS}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_version = version_dataset(records, name=mode, out_dir=out, sources=sources)

    manifest = TrainingJobManifest(
        job_id=f"{mode}-{dataset_version.version}-{_now_ms()}",
        mode=mode,
        orchestrator=orchestrator,
        dataset=dataset_version.to_dict(),
        base_ref=base_ref,
        checkpoint_plan={
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "resume_from": resume_from,
        },
        telemetry_sink=str(out / f"{mode}-{dataset_version.version}.telemetry.json"),
        hyperparameters=dict(hyperparameters or {}),
        created_at_ms=_now_ms(),
    )
    (out / "training-job.manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    )
    return manifest, dataset_version


__all__ = [
    "CheckpointEntry",
    "CheckpointRegistry",
    "DatasetVersion",
    "LossTelemetry",
    "ORCHESTRATORS",
    "TRAINING_MODES",
    "TrainingJobManifest",
    "build_training_job_manifest",
    "compute_dataset_digest",
    "load_dataset_manifest",
    "parse_trainer_log",
    "records_for_mode",
    "telemetry_from_loss_history",
    "telemetry_from_summary",
    "telemetry_to_training_health",
    "version_dataset",
]
