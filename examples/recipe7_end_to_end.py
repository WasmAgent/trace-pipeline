"""examples/recipe7_end_to_end.py — full audit in one command.

This is just a thin wrapper around run_audit.py — the real script lives
at the repo root because it's the headline reproducer.

Run from repo root:
    python examples/recipe7_end_to_end.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

result = subprocess.run(
    [sys.executable, str(ROOT / "run_audit.py")],
    cwd=ROOT,
    check=False,
)
sys.exit(result.returncode)
