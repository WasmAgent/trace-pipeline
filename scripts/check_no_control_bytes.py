#!/usr/bin/env python3
"""
check-no-control-bytes.py

Scan {eval_trust, evomerge, scripts, tests, examples, run_audit.py}
for NUL (0x00) and other C0 control bytes (excluding tab/LF/CR) inside
Python source files.

Why this exists: a NUL byte slipped into a regex character class in the
wasmagent-js sibling repo on 2026-06-26, surviving git commit and tests
but breaking awk/grep/file reporting. The same class of bug can land in
Python too (less likely because Python rejects most control bytes at
parse time, but pyc cache and string literals can hide them).

Run by:
  - .githooks/pre-push
  - CI (.github/workflows/*)

Exit 0 = clean, exit 1 = at least one offending file.
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["eval_trust", "evomerge", "scripts", "tests", "examples"]
EXTRA_FILES = ["run_audit.py"]
EXT = {".py"}
IGNORE_DIRS = {"__pycache__", ".venv", "venv", ".git", "dist", "build", "node_modules"}

BAD_BYTES = set(range(0x00, 0x20)) - {0x09, 0x0a, 0x0d}
BAD_BYTES.add(0x7f)


def describe(b: int) -> str:
    names = {0x00: "NUL", 0x07: "BEL", 0x08: "BS", 0x0b: "VT", 0x0c: "FF",
             0x1b: "ESC", 0x7f: "DEL"}
    return f"{names.get(b, '')} (\\x{b:02x})".lstrip()


def line_col(data: bytes, offset: int) -> tuple[int, int]:
    line = data.count(b"\n", 0, offset) + 1
    last_nl = data.rfind(b"\n", 0, offset)
    col = offset - last_nl if last_nl >= 0 else offset + 1
    return line, col


def walk(root: Path):
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in EXT:
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        yield p


def main() -> int:
    errors = 0
    scanned = 0
    for tgt in TARGETS:
        for path in walk(REPO_ROOT / tgt):
            scanned += 1
            data = path.read_bytes()
            for i, b in enumerate(data):
                if b in BAD_BYTES:
                    line, col = line_col(data, i)
                    rel = path.relative_to(REPO_ROOT)
                    print(f"{rel}:{line}:{col}  {describe(b)} byte at offset {i}",
                          file=sys.stderr)
                    errors += 1
                    break
    for extra in EXTRA_FILES:
        path = REPO_ROOT / extra
        if not path.is_file():
            continue
        scanned += 1
        data = path.read_bytes()
        for i, b in enumerate(data):
            if b in BAD_BYTES:
                line, col = line_col(data, i)
                print(f"{extra}:{line}:{col}  {describe(b)} byte at offset {i}",
                      file=sys.stderr)
                errors += 1
                break

    if errors:
        print(f"\n✗ {errors} file(s) contain disallowed control bytes.",
              file=sys.stderr)
        print("  Use \\uXXXX or \\xXX escape sequences in regex literals.",
              file=sys.stderr)
        return 1

    print(f"✓ No disallowed control bytes in {scanned} source files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
