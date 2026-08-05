"""Run the repository quality gate on Modal.

Usage:
    modal run scripts/modal_gate.py
    modal run scripts/modal_gate.py --paths tests/test_eval_sandbox.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import TypedDict

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"
_TAIL_CHARS = 6000
_PYTEST_COUNT = re.compile(r"\b(\d+)\s+(passed|failed|errors?)\b")
_PYTEST_DURATION = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")


class GateReport(TypedDict):
    """Machine-readable result from one remote quality gate."""

    ruff_ok: bool
    pytest_ok: bool
    passed: int
    failed: int
    errors: int
    ruff_tail: str
    pytest_tail: str


def _head_sha() -> str:
    """Return the exact local commit that the Modal image must contain.

    Module level code runs twice: once locally to build the image, and again
    inside the container when Modal imports this file. There is no git
    repository in the container, so an unguarded call crash-loops every worker
    before it starts. The commit is only needed locally anyway, because the
    container already holds the code the build step checked out.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return result.stdout.strip() or "main"


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
    )
)

app = modal.App("autoevolve-gate")


def _output_tail(completed: subprocess.CompletedProcess[str]) -> str:
    """Keep the end of both captured streams without dropping either one."""

    parts: list[str] = []
    if completed.stdout:
        parts.append("stdout:\n" + completed.stdout[-_TAIL_CHARS:])
    if completed.stderr:
        parts.append("stderr:\n" + completed.stderr[-_TAIL_CHARS:])
    return "\n".join(parts)


def _pytest_counts(stdout: str, stderr: str) -> tuple[int, int, int]:
    """Parse passed, failed, and error counts from pytest's summary line."""

    output = "\n".join((stdout, stderr))
    for line in reversed(output.splitlines()):
        if not _PYTEST_DURATION.search(line):
            continue
        counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0}
        matches = _PYTEST_COUNT.findall(line)
        if not matches and "no tests ran" not in line:
            continue
        for raw_count, label in matches:
            counts[label] = int(raw_count)
        return counts["passed"], counts["failed"], counts["error"] + counts["errors"]
    # Pytest usage and internal failures can exit before printing a test summary.
    # Do not invent counts. The nonzero status and captured tail remain the verdict.
    return 0, 0, 0


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one gate command and preserve nonzero exits as ordinary results."""

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@app.function(image=image, cpu=4.0, memory=8192, timeout=60 * 30)
def gate(paths: str | None = None) -> GateReport:
    """Run ruff and pytest once, optionally narrowing pytest to selected paths."""

    ruff = _run(["uv", "run", "ruff", "check", "."])
    pytest_command = ["uv", "run", "pytest", "-q"]
    if paths:
        pytest_command.extend(shlex.split(paths))
    pytest = _run(pytest_command)
    passed, failed, errors = _pytest_counts(pytest.stdout, pytest.stderr)
    return {
        "ruff_ok": ruff.returncode == 0,
        "pytest_ok": pytest.returncode == 0,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "ruff_tail": _output_tail(ruff),
        "pytest_tail": _output_tail(pytest),
    }


@app.local_entrypoint()
def main(paths: str | None = None) -> None:
    """Print the remote report and expose its verdict as the process exit code."""

    report = gate.remote(paths)
    print(json.dumps(report, indent=2))
    if not report["ruff_ok"] or not report["pytest_ok"]:
        raise SystemExit(1)
