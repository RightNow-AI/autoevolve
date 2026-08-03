import random
import subprocess
from pathlib import Path

import pytest

from autoevolve.core.types import Budget, Contract, EvalOutcome, ParentBundle, Program
from autoevolve.mutate import agentic
from autoevolve.mutate.agentic import AgenticOperator
from autoevolve.mutate.base import OperatorContext, OperatorError


def _bundle() -> ParentBundle:
    parent = Program("p1", "r1", None, "seed", "ref", 0, None, "now")
    return ParentBundle(
        parent,
        {"main.py": "# EVOLVE-BLOCK-START\nvalue = 1\n# EVOLVE-BLOCK-END\n"},
        discoveries=["Larger batches improved throughput on the same fixture set."],
        operator_hint="The previous candidate failed parity fixture 2.",
    )


def _context(tmp_path: Path, evaluator) -> OperatorContext:
    return OperatorContext(
        contract=Contract(
            "increase value",
            "general",
            "score",
            True,
            1.0,
            2.0,
            "correct",
            Budget(max_evals=10),
        ),
        rng=random.Random(1),
        endpoint_cheap=None,
        endpoint_strong=None,
        evaluate_locally=evaluator,
        workdir=tmp_path,
    )


def test_agentic_reads_back_edit_and_appends_local_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOEVOLVE_AGENT_RUNTIME", "claude")
    monkeypatch.setattr(agentic.shutil, "which", lambda name: f"C:/bin/{name}.exe")
    calls: list[tuple[list[str], Path, float]] = []
    evaluations: list[dict[str, str]] = []

    def fake_process(
        cmd: list[str], cwd: Path, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, cwd, timeout_s))
        (cwd / "main.py").write_text(
            "# EVOLVE-BLOCK-START\nvalue = 2\n# EVOLVE-BLOCK-END\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    def evaluate(files: dict[str, str]) -> EvalOutcome:
        evaluations.append(files)
        return EvalOutcome(True, {"score": 2.0, "correct": 1.0}, 0)

    monkeypatch.setattr(agentic, "run_agent_process", fake_process)

    proposal = AgenticOperator().propose(_bundle(), _context(tmp_path, evaluate))

    workspace = tmp_path / "agentic-p1"
    assert calls == [
        (
            [
                "claude",
                "-p",
                agentic.AGENT_TASK_PROMPT,
                "--permission-mode",
                "acceptEdits",
                "--allowedTools",
                "Read",
                "Edit",
                "Write",
                "--max-turns",
                "12",
                "--output-format",
                "json",
            ],
            workspace,
            600.0,
        )
    ]
    assert proposal.files["main.py"].splitlines()[1] == "value = 2"
    assert evaluations == [proposal.files]
    assert "runtime=claude changed=1" in proposal.notes
    assert '"gate_passed":true' in proposal.notes
    assert "Recent failure notes" in (workspace / "PROMPT.md").read_text(encoding="utf-8")


def test_agentic_raises_when_agent_makes_no_change(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOEVOLVE_AGENT_RUNTIME", "codex")
    monkeypatch.setattr(agentic.shutil, "which", lambda name: f"C:/bin/{name}.exe")

    def fake_process(
        cmd: list[str], cwd: Path, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(agentic, "run_agent_process", fake_process)

    with pytest.raises(OperatorError, match="agent made no changes"):
        AgenticOperator().propose(
            _bundle(),
            _context(tmp_path, lambda files: EvalOutcome(True, {"score": 1.0}, 0)),
        )


def test_runtime_selection_respects_explicit_and_auto_modes(monkeypatch):
    monkeypatch.setenv("AUTOEVOLVE_AGENT_RUNTIME", "codex")
    monkeypatch.setattr(
        agentic.shutil, "which", lambda name: "C:/bin/codex.exe" if name == "codex" else None
    )
    assert agentic._select_runtime() == "codex"

    monkeypatch.setenv("AUTOEVOLVE_AGENT_RUNTIME", "auto")
    monkeypatch.setattr(agentic.shutil, "which", lambda name: f"C:/bin/{name}.exe")
    assert agentic._select_runtime() == "claude"

    monkeypatch.setattr(
        agentic.shutil, "which", lambda name: "C:/bin/codex.exe" if name == "codex" else None
    )
    assert agentic._select_runtime() == "codex"


def test_verified_claude_and_codex_commands_are_exact(tmp_path):
    assert agentic._agent_command("claude", tmp_path) == [
        "claude",
        "-p",
        agentic.AGENT_TASK_PROMPT,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Read",
        "Edit",
        "Write",
        "--max-turns",
        "12",
        "--output-format",
        "json",
    ]
    assert agentic._agent_command("codex", tmp_path) == [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-s",
        "workspace-write",
        "-C",
        str(tmp_path),
        "-o",
        str(tmp_path / "last-message.txt"),
        agentic.AGENT_TASK_PROMPT,
    ]
