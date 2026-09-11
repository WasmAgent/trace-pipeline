"""Validated filesystem paths for operator-supplied output locations.

Every command that writes a file whose location comes from an argument or
caller routes that location through :func:`output_path`, so the checks live
in one audited place instead of being re-derived (or skipped) per call site.
"""
from __future__ import annotations

from pathlib import Path, PurePath

_CONTROL_CHARS = frozenset(chr(c) for c in range(32)) | {chr(127)}


def output_path(raw: str | PurePath) -> Path:
    """Return ``raw`` as a resolved, validated write target.

    - rejects empty strings and NUL / control characters (path confusion);
    - rejects ``..`` segments, so a caller can never smuggle a traversal past
      a base-directory contract such as ``base_dir / output_path(arg)``;
    - resolves symlinks and ``.``/``..`` to an absolute final target, making
      the destination unambiguous in logs and audits.

    Operator-facing CLIs legitimately accept absolute destinations, so the
    contract here is validation + resolution, not restriction to a base
    directory.
    """
    if isinstance(raw, PurePath):
        raw = str(raw)
    if not raw:
        raise ValueError("output path must not be empty")
    if "\x00" in raw or any(ch in _CONTROL_CHARS for ch in raw):
        raise ValueError(f"output path contains control characters: {raw!r}")
    if ".." in Path(raw).parts:
        raise ValueError(f"output path must not contain '..' segments: {raw!r}")
    return Path(raw).expanduser().resolve()
