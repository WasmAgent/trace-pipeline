"""Benchmark linter v0.1 — anti-reward-hacking exploit surface detection.

Checks a benchmark task directory for risks that would allow an agent to
achieve high scores without actually solving the task.

Usage:
    from evomerge.security.benchmark_linter import lint_benchmark_dir, print_lint_report

    findings = lint_benchmark_dir(Path("benchmarks/my-task"))
    print_lint_report(findings)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Severity = Literal["critical", "high", "medium", "low", "info"]
Surface = Literal["gold_answer", "test_script", "env_leak", "grader", "network", "git_log", "unknown"]


@dataclass
class LintFinding:
    check_id: str
    severity: Severity
    surface: Surface
    description: str
    path: str | None = None


@dataclass
class BenchmarkTrustScore:
    task_path: str
    findings: list[LintFinding] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def score(self) -> float:
        """0.0 = fully exploitable, 1.0 = clean. Deduct per finding severity."""
        deductions = sum({
            "critical": 0.4, "high": 0.2, "medium": 0.1, "low": 0.05, "info": 0.0
        }[f.severity] for f in self.findings)
        return max(0.0, 1.0 - deductions)

    @property
    def trusted(self) -> bool:
        return self.critical_count == 0 and self.high_count == 0


# ── Checks ────────────────────────────────────────────────────────────────────

_GOLD_ANSWER_NAMES = {
    "answer.txt", "gold.txt", "gold_answer.txt", "expected_output.txt",
    "solution.txt", "correct.txt", "expected.txt", "answer.json", "gold.json",
}

_GRADER_NAMES = {
    "grade.py", "grader.py", "grader.sh", "eval.py", "evaluator.py",
    "score.py", "check.py", "checker.py", "verify.py", "run_eval.sh",
}


def check_gold_answer_leak(root: Path) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in _GOLD_ANSWER_NAMES:
            try:
                readable = os.access(p, os.R_OK)
            except OSError:
                readable = False
            if readable:
                findings.append(LintFinding(
                    check_id="gold-answer-readable",
                    severity="critical",
                    surface="gold_answer",
                    description=f"Gold answer file is readable by the agent: {p.name}",
                    path=str(p.relative_to(root)),
                ))
    return findings


def check_test_script_writability(root: Path) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for p in root.rglob("*.py"):
        if p.is_file() and "test" in p.name.lower():
            try:
                writable = os.access(p, os.W_OK)
            except OSError:
                writable = False
            if writable:
                findings.append(LintFinding(
                    check_id="test-script-writable",
                    severity="high",
                    surface="test_script",
                    description=f"Test script is writable (agent could monkey-patch): {p.name}",
                    path=str(p.relative_to(root)),
                ))
    return findings


def check_env_leak(root: Path) -> list[LintFinding]:
    findings: list[LintFinding] = []
    # Check for .env files in task root
    for p in root.glob(".env*"):
        if p.is_file():
            findings.append(LintFinding(
                check_id="env-file-present",
                severity="high",
                surface="env_leak",
                description=f"Environment file present in task root: {p.name}",
                path=str(p.relative_to(root)),
            ))
    # Check for ANSWER/SOLUTION/GOLD env vars in any shell scripts
    for p in root.rglob("*.sh"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for keyword in ("ANSWER=", "SOLUTION=", "GOLD=", "EXPECTED="):
            if keyword in text:
                findings.append(LintFinding(
                    check_id="env-answer-in-script",
                    severity="medium",
                    surface="env_leak",
                    description=f"Script contains answer-like variable {keyword!r}",
                    path=str(p.relative_to(root)),
                ))
                break
    return findings


def check_grader_replaceability(root: Path) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in _GRADER_NAMES:
            try:
                writable = os.access(p, os.W_OK)
                mode = p.stat().st_mode
            except OSError:
                continue
            if writable:
                findings.append(LintFinding(
                    check_id="grader-writable",
                    severity="critical",
                    surface="grader",
                    description=f"Grader script is writable (could be replaced): {p.name}",
                    path=str(p.relative_to(root)),
                ))
    return findings


def check_git_log_leak(root: Path) -> list[LintFinding]:
    findings: list[LintFinding] = []
    git_dir = root / ".git"
    if git_dir.exists() and git_dir.is_dir():
        findings.append(LintFinding(
            check_id="git-dir-present",
            severity="high",
            surface="git_log",
            description="Task directory contains .git — git log may expose the solution",
            path=".git",
        ))
    return findings


def check_network_policy(root: Path) -> list[LintFinding]:
    """Check if task expects network isolation but has no manifest declaring it."""
    findings: list[LintFinding] = []
    has_manifest = any(root.glob("task.json")) or any(root.glob("task.yaml")) or any(root.glob("task.yml"))
    if not has_manifest:
        findings.append(LintFinding(
            check_id="no-task-manifest",
            severity="info",
            surface="network",
            description="No task.json/yaml manifest found — network isolation policy is undeclared",
        ))
    return findings


# ── Public API ────────────────────────────────────────────────────────────────

def lint_benchmark_dir(path: Path) -> BenchmarkTrustScore:
    """Run all checks on a benchmark task directory and return a BenchmarkTrustScore."""
    if not path.is_dir():
        return BenchmarkTrustScore(
            task_path=str(path),
            findings=[LintFinding(
                check_id="path-not-found",
                severity="critical",
                surface="unknown",
                description=f"Task directory not found: {path}",
            )],
        )
    findings: list[LintFinding] = []
    findings.extend(check_gold_answer_leak(path))
    findings.extend(check_test_script_writability(path))
    findings.extend(check_env_leak(path))
    findings.extend(check_grader_replaceability(path))
    findings.extend(check_git_log_leak(path))
    findings.extend(check_network_policy(path))
    return BenchmarkTrustScore(task_path=str(path), findings=findings)


def print_lint_report(result: BenchmarkTrustScore) -> None:
    status = "TRUSTED" if result.trusted else "UNTRUSTED"
    print(f"[{status}] {result.task_path}  trust_score={result.score:.2f}")
    for f in result.findings:
        path_str = f"  ({f.path})" if f.path else ""
        print(f"  [{f.severity.upper():8}] {f.check_id}: {f.description}{path_str}")
    if not result.findings:
        print("  No findings.")
