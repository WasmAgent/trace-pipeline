"""Tests for benchmark linter v0.1."""
from __future__ import annotations

import tempfile
from pathlib import Path


from evomerge.security.benchmark_linter import (
    lint_benchmark_dir,
)


def make_task(files: dict[str, str]) -> Path:
    """Create a temp dir with the given files (name -> content)."""
    tmp = Path(tempfile.mkdtemp())
    for name, content in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return tmp


def test_clean_task():
    d = make_task({"task.json": '{"name":"test"}', "src/main.py": "print('hello')"})
    result = lint_benchmark_dir(d)
    assert result.trusted
    assert result.score == 1.0


def test_gold_answer_detected():
    d = make_task({"answer.txt": "42", "task.json": "{}"})
    result = lint_benchmark_dir(d)
    ids = [f.check_id for f in result.findings]
    assert "gold-answer-readable" in ids
    critical = [f for f in result.findings if f.severity == "critical"]
    assert len(critical) >= 1


def test_git_dir_detected():
    d = make_task({"task.json": "{}"})
    git_dir = d / ".git"
    git_dir.mkdir()
    result = lint_benchmark_dir(d)
    ids = [f.check_id for f in result.findings]
    assert "git-dir-present" in ids


def test_no_manifest_info_finding():
    d = make_task({"src/main.py": "pass"})
    result = lint_benchmark_dir(d)
    ids = [f.check_id for f in result.findings]
    assert "no-task-manifest" in ids


def test_path_not_found():
    result = lint_benchmark_dir(Path("/nonexistent/task/dir/xyz"))
    assert not result.trusted
    assert result.findings[0].check_id == "path-not-found"


def test_env_file_detected():
    d = make_task({"task.json": "{}", ".env": "ANSWER=secret"})
    result = lint_benchmark_dir(d)
    ids = [f.check_id for f in result.findings]
    assert "env-file-present" in ids


def test_trust_score_decreases_with_findings():
    d = make_task({"answer.txt": "42"})  # no manifest either
    result = lint_benchmark_dir(d)
    assert result.score < 1.0
