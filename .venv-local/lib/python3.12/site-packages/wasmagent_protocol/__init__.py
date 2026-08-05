"""wasmagent-protocol — canonical AEP + compliance JSON Schemas.

Single source of truth across the WasmAgent org. Do not copy these schemas into
consumer repositories; depend on this package instead.

    from wasmagent_protocol import get_schema, schema_path, INDEX

    aep = get_schema("aep-record")   # parsed dict
    path = schema_path("aep-record") # pathlib.Path to the .json file
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["INDEX", "get_schema", "schema_path", "schema_ids"]

_SCHEMAS_PKG = "wasmagent_protocol.schemas"
# Editable / source-checkout fallback: the canonical schemas live at the repo
# root under schemas/, which is two directories above this file
# (src/wasmagent_protocol/__init__.py -> <repo>/schemas). Built wheels ship the
# same tree inside the package via force-include, so the resource API is tried
# first; this Path is the fallback for `pip install -e .` where force-include is
# not materialized.
_SOURCE_SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def _schemas_root():
    """Locate the canonical schemas directory as a Traversable.

    Installed wheels package schemas under ``wasmagent_protocol.schemas``
    (force-include in pyproject). Editable installs do not materialize that
    mapping, so fall back to the repo-root ``schemas/`` directory.
    """
    try:
        root = resources.files(_SCHEMAS_PKG)
        if root.joinpath("index.json").is_file():
            return root
    except Exception:  # pragma: no cover - depends on install layout
        pass
    if (_SOURCE_SCHEMAS / "index.json").is_file():
        return _SOURCE_SCHEMAS
    raise FileNotFoundError(
        "wasmagent_protocol schemas not found: neither the packaged "
        "wasmagent_protocol.schemas resource nor the source checkout at "
        f"{_SOURCE_SCHEMAS} contains index.json"
    )


def _read(rel_path: str) -> str:
    # rel_path is e.g. "index.json" or "aep/aep-record.schema.json"
    node = _schemas_root()
    for part in rel_path.split("/"):
        node = node.joinpath(part)
    return node.read_text(encoding="utf-8")


INDEX: dict[str, Any] = json.loads(_read("index.json"))

# Map schema id -> path relative to the schemas/ directory.
_PATHS: dict[str, str] = {
    s["id"]: s["path"].removeprefix("schemas/") for s in INDEX["schemas"]
}


def schema_ids() -> list[str]:
    """Return the ids of every registered canonical schema."""
    return list(_PATHS)


@lru_cache(maxsize=None)
def get_schema(schema_id: str) -> dict[str, Any]:
    """Return the parsed JSON Schema for a registered id.

    Raises KeyError on an unknown id.
    """
    try:
        rel = _PATHS[schema_id]
    except KeyError:
        raise KeyError(
            f"unknown schema id {schema_id!r}; known: {', '.join(_PATHS)}"
        ) from None
    return json.loads(_read(rel))


def schema_path(schema_id: str) -> Path:
    """Return a filesystem Path to the schema JSON for a registered id.

    Raises KeyError on an unknown id.
    """
    try:
        rel = _PATHS[schema_id]
    except KeyError:
        raise KeyError(
            f"unknown schema id {schema_id!r}; known: {', '.join(_PATHS)}"
        ) from None
    node = _schemas_root()
    for part in rel.split("/"):
        node = node.joinpath(part)
    if isinstance(node, Path):
        return node
    with resources.as_file(node) as p:
        return Path(p)
