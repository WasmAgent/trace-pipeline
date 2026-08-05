"""Cross-repository schema drift detection.

A vendored copy of a canonical ``wasmagent.dev`` schema is *drifted* when its
normalized JSON differs from the canonical schema shipped by this package at the
pinned version. This module implements the comparison and a repo-wide scan that
the published ``wasmagent-protocol check`` CLI and the shared
``schema-drift.yml`` workflow both build on.

Three failure modes are detected:

- ``drift`` — a ``*.schema.json`` re-declares a canonical ``$id`` but its
  content differs from the canonical schema.
- ``no-package-dep`` — a file re-declares a canonical ``$id`` while the repo
  does not depend on ``@wasmagent/protocol`` / ``wasmagent-protocol`` (a fork
  with no tracking).
- ``competing-registry`` — a repo other than ``wasmagent-protocol`` ships a
  ``schemas/index.json`` listing canonical schema ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import INDEX, get_schema

CANONICAL_HOST = INDEX.get("canonical_host", "https://wasmagent.dev/schemas/")

_PACKAGE_NAMES = {"@wasmagent/protocol", "wasmagent-protocol"}

# Directories never worth scanning for vendored schemas or competing registries.
_SCAN_IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".claude-bot",
    ".turbo",
    ".next",
    "target",
    "tests",
    ".idea",
    ".vscode",
}


def normalize_schema(obj: Any) -> str:
    """Canonical, order-independent serialization of a schema document.

    Accepts either a raw JSON string (a vendored file) or an already-parsed
    object (the canonical schema returned by ``get_schema``). Two schemas that
    differ only in key order or whitespace compare equal.
    """
    if isinstance(obj, str):
        obj = json.loads(obj)
    return json.dumps(_sort_keys(obj), separators=(",", ":"))


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


def canonical_ids() -> set[str]:
    """Return the set of canonical ``$id`` URIs for every registered schema."""
    return {entry["canonical_id"] for entry in INDEX["schemas"]}


def schema_id_for_canonical(canonical_id: str) -> str | None:
    """Map a canonical ``$id`` back to its registry id, or ``None``."""
    for entry in INDEX["schemas"]:
        if entry["canonical_id"] == canonical_id:
            return entry["id"]
    return None


@dataclass
class Finding:
    """A single drift-scan result.

    ``severity`` is ``"error"`` (blocks CI), ``"ok"`` (canonical match), or
    ``"skip"`` (not a canonical schema / not applicable).
    """

    severity: str
    code: str
    path: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


def _ok(code: str, path: Path, message: str) -> Finding:
    return Finding("ok", code, str(path), message)


def _error(code: str, path: Path, message: str) -> Finding:
    return Finding("error", code, str(path), message)


def check_file(path: str | Path, schema_id: str) -> Finding:
    """Compare a single vendored schema file to the canonical ``schema_id``.

    Raises ``FileNotFoundError`` if ``path`` does not exist and ``KeyError`` if
    ``schema_id`` is not a registered canonical id.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    canonical = get_schema(schema_id)  # raises KeyError on unknown id
    if normalize_schema(text) == normalize_schema(canonical):
        return _ok("match", p, f"{schema_id}: matches canonical")
    return _error(
        "drift",
        p,
        f"{schema_id}: vendored schema differs from canonical "
        f"({INDEX.get('protocol', 'AEP')} package)",
    )


def _iter_files(root: Path, suffix: str):
    for path in root.rglob(f"*{suffix}"):
        if any(part in _SCAN_IGNORE_DIRS for part in path.parts):
            continue
        yield path


def is_canonical_source(root: str | Path) -> bool:
    """True if ``root`` is the ``wasmagent-protocol`` repo itself."""
    root = Path(root)
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            if json.loads(pkg.read_text(encoding="utf-8")).get("name") in _PACKAGE_NAMES:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    pyproj = root / "pyproject.toml"
    if pyproj.is_file():
        text = pyproj.read_text(encoding="utf-8")
        if 'name = "wasmagent-protocol"' in text or "name = 'wasmagent-protocol'" in text:
            return True
    return False


def depends_on_package(root: str | Path) -> bool:
    """True if ``root`` declares a dependency on the protocol package.

    Best-effort heuristic over npm ``package.json`` dependency maps and Python
    ``requirements*.txt`` / ``pyproject.toml``.
    """
    root = Path(root)
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            deps = data.get(key)
            if isinstance(deps, dict) and any(name in _PACKAGE_NAMES for name in deps):
                return True
    for req_name in ("requirements.txt", "requirements-dev.txt", "requirements.txt.in"):
        req = root / req_name
        if req.is_file() and "wasmagent-protocol" in req.read_text(encoding="utf-8"):
            return True
    pyproj = root / "pyproject.toml"
    if pyproj.is_file():
        # Only count it as a dependency if the project is NOT the package itself.
        text = pyproj.read_text(encoding="utf-8")
        if (
            'name = "wasmagent-protocol"' not in text
            and "name = 'wasmagent-protocol'" not in text
            and "wasmagent-protocol" in text
        ):
            return True
    return False


def scan(
    root: str | Path,
    *,
    allow_canonical_source: bool = False,
) -> list[Finding]:
    """Scan ``root`` for vendored canonical-schema drift and competing registries.

    When ``root`` is the ``wasmagent-protocol`` repo (auto-detected, or forced
    via ``allow_canonical_source``), the no-package-dep and competing-registry
    checks are skipped — this repo is the legitimate source of both.
    """
    root = Path(root).resolve()
    treat_as_source = is_canonical_source(root) or allow_canonical_source
    has_dep = depends_on_package(root)

    findings: list[Finding] = []
    canonical_by_id = {entry["canonical_id"]: entry for entry in INDEX["schemas"]}

    for path in _iter_files(root, ".schema.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(_error("invalid-json", path, f"cannot parse JSON: {exc}"))
            continue
        sid = doc.get("$id") if isinstance(doc, dict) else None
        if not isinstance(sid, str) or not sid.startswith(CANONICAL_HOST):
            continue  # not a canonical schema — leave it alone
        entry = canonical_by_id.get(sid)
        if entry is None:
            findings.append(
                _error(
                    "unknown-canonical-id",
                    path,
                    f"$id {sid} is under the canonical host but is not registered",
                )
            )
            continue
        if normalize_schema(doc) != normalize_schema(get_schema(entry["id"])):
            findings.append(
                _error(
                    "drift",
                    path,
                    f"{entry['id']}: vendored schema differs from canonical",
                )
            )
            continue
        # Content matches the canonical schema.
        if treat_as_source:
            findings.append(_ok("match", path, f"{entry['id']}: matches canonical (source repo)"))
        elif has_dep:
            findings.append(_ok("match", path, f"{entry['id']}: matches canonical"))
        else:
            findings.append(
                _error(
                    "no-package-dep",
                    path,
                    f"{entry['id']}: re-declares canonical $id but the repo does not "
                    "depend on @wasmagent/protocol / wasmagent-protocol",
                )
            )

    # Registry guard: only wasmagent-protocol may ship a canonical registry.
    if not treat_as_source:
        for idx_path in root.rglob("index.json"):
            if any(part in _SCAN_IGNORE_DIRS for part in idx_path.parts):
                continue
            try:
                idx = json.loads(idx_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            entries = idx.get("schemas") if isinstance(idx, dict) else None
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                cid = entry.get("canonical_id")
                if isinstance(cid, str) and cid.startswith(CANONICAL_HOST):
                    findings.append(
                        _error(
                            "competing-registry",
                            idx_path,
                            f"lists canonical schema id {cid} — only the "
                            "wasmagent-protocol repo may ship a canonical registry",
                        )
                    )
                    break

    return findings


def has_drift(findings: list[Finding]) -> bool:
    return any(f.is_error for f in findings)
