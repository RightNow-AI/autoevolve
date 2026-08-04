"""Headless Claude Code or Codex mutation operator.

Claude CLI reference: https://code.claude.com/docs/en/cli-reference
Codex command flags were verified from codex-cli 0.146.0 help output.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from autoevolve.core.types import ParentBundle, Proposal
from autoevolve.mutate.base import OperatorContext, OperatorError

AGENT_TASK_PROMPT = "read PROMPT.md and improve the code per its contract"


def run_agent_process(
    cmd: list[str], cwd: Path, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    """Run the selected headless coding agent through one patchable seam."""

    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout_s,
        check=False,
        capture_output=True,
        text=True,
    )


class AgenticOperator:
    """Let a headless coding agent edit a scratch copy of the parent."""

    name = "agentic"

    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal:
        runtime, executable = _select_runtime()
        timeout_s = _agent_timeout()
        # Unique per cycle. Reusing one directory per parent meant a retry had
        # to delete a tree the previous agent might still hold open.
        workspace = ctx.workdir / f"agentic-{bundle.parent.id}-{uuid.uuid4().hex[:8]}"
        _prepare_workspace(workspace, ctx.workdir)

        for relative_path, content in bundle.parent_files.items():
            target = _safe_path(workspace, relative_path)
            if target.name == "PROMPT.md" and target.parent == workspace:
                raise OperatorError("parent file path PROMPT.md conflicts with the agent contract")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (workspace / "PROMPT.md").write_text(
            _agent_contract(bundle, ctx), encoding="utf-8"
        )

        command = _agent_command(runtime, workspace, executable)
        try:
            completed = run_agent_process(command, workspace, timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise OperatorError(f"{runtime} agent timed out after {timeout_s:g} seconds") from exc
        except OSError as exc:
            raise OperatorError(f"failed to start {runtime} agent: {exc}") from exc

        # The edit on disk is the mutation. The exit code describes the agent
        # session, which ends with teardown this operator does not care about:
        # every agentic cycle of the first real run finished its edit and then
        # exited 1 because a SessionEnd hook in the host's plugin config could
        # not resolve, and good work was thrown away each time. A nonzero exit
        # is recorded in the notes and left for the gate to judge, because the
        # gate reads the code and the exit code does not.
        files: dict[str, str] = {}
        changed = 0
        missing: list[str] = []
        for relative_path, original in bundle.parent_files.items():
            target = _safe_path(workspace, relative_path)
            if not target.is_file():
                missing.append(relative_path)
                continue
            content = target.read_text(encoding="utf-8")
            files[relative_path] = content
            changed += content != original
        detail = (completed.stderr or completed.stdout or "no output").strip()[:500]
        if missing:
            raise OperatorError(
                f"{runtime} agent removed parent files: {', '.join(sorted(missing))}"
            )
        if changed == 0:
            if completed.returncode != 0:
                raise OperatorError(
                    f"{runtime} agent exited {completed.returncode} without "
                    f"changing anything: {detail}"
                )
            raise OperatorError("agent made no changes")

        notes = [f"agentic: runtime={runtime} changed={changed}"]
        if completed.returncode != 0:
            notes.append(f"agent_exit={completed.returncode}")
        if ctx.evaluate_locally is not None:
            outcome = ctx.evaluate_locally(files)
            notes.append(
                "local_eval="
                + json.dumps(
                    {
                        "gate_passed": outcome.gate_passed,
                        "scores": outcome.scores,
                        "stage_reached": outcome.stage_reached,
                        "error": outcome.error,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return Proposal(files=files, notes=" ".join(notes))


def _select_runtime() -> tuple[str, str]:
    """Return the runtime name and the executable path to spawn.

    The resolved path matters. shutil.which honors PATHEXT and happily finds
    codex.cmd, but CreateProcess only appends .exe, so spawning the bare name
    raises FileNotFoundError. That surfaced as a skipped cycle rather than an
    error, which is why every agentic mutation silently failed on Windows npm
    installs while the preflight check reported the runtime as present.
    """

    requested = os.getenv("AUTOEVOLVE_AGENT_RUNTIME", "auto").strip().lower()
    if requested not in {"auto", "claude", "codex"}:
        raise OperatorError("AUTOEVOLVE_AGENT_RUNTIME must be claude, codex, or auto")
    if requested in {"claude", "codex"}:
        resolved = shutil.which(requested)
        if resolved is None:
            raise OperatorError(
                f"AUTOEVOLVE_AGENT_RUNTIME={requested}, but {requested} is not on PATH"
            )
        return requested, resolved
    for name in ("claude", "codex"):
        resolved = shutil.which(name)
        if resolved:
            return name, resolved
    raise OperatorError("no agent runtime found; install claude or codex")


#: Execution is the whole point of this operator. Without it the agent cannot
#: measure anything, and on a search problem it has nothing to reason from: a
#: probe on Golomb order 29 spent all twelve of its turns having seven Bash and
#: PowerShell calls denied, hit the turn limit, and never made an edit.
#:
#: The trust boundary is worth stating plainly. This agent is engine side. It
#: runs with the engine's privileges, exactly like the model operators, and it
#: cannot certify its own work: the engine re-runs the whole cascade in the
#: sandbox on the files the operator returns, so an agent's own evaluation is a
#: note and never a score. What execution does change is that a campaign pack's
#: goal text, which reaches the agent's prompt, becomes actionable. Run packs
#: you trust, or narrow this list. See SECURITY.md.
_DEFAULT_AGENT_TOOLS = ("Read", "Edit", "Write", "Bash")
_DEFAULT_AGENT_TURNS = 40


def _agent_tools() -> tuple[str, ...]:
    """Tools the mutation agent may use, overridable for untrusted packs."""

    raw = os.getenv("AUTOEVOLVE_AGENTIC_TOOLS", "").strip()
    if not raw:
        return _DEFAULT_AGENT_TOOLS
    tools = tuple(name.strip() for name in raw.split(",") if name.strip())
    if not tools:
        raise OperatorError("AUTOEVOLVE_AGENTIC_TOOLS must name at least one tool")
    return tools


def _agent_turns() -> int:
    """Turn budget for one mutation session.

    Twelve was not enough to survive a few denied calls, let alone write code,
    run it, read the result, and revise.
    """

    raw = os.getenv("AUTOEVOLVE_AGENTIC_TURNS", str(_DEFAULT_AGENT_TURNS))
    try:
        turns = int(raw)
    except ValueError as exc:
        raise OperatorError("AUTOEVOLVE_AGENTIC_TURNS must be a positive integer") from exc
    if turns <= 0:
        raise OperatorError("AUTOEVOLVE_AGENTIC_TURNS must be a positive integer")
    return turns


def _agent_timeout() -> float:
    raw = os.getenv("AUTOEVOLVE_AGENTIC_TIMEOUT_S", "600")
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise OperatorError("AUTOEVOLVE_AGENTIC_TIMEOUT_S must be a positive number") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise OperatorError("AUTOEVOLVE_AGENTIC_TIMEOUT_S must be a positive number")
    return timeout_s


def _agent_command(runtime: str, workspace: Path, executable: str | None = None) -> list[str]:
    program = executable or runtime
    if runtime == "claude":
        return [
            program,
            "-p",
            AGENT_TASK_PROMPT,
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            *_agent_tools(),
            "--max-turns",
            str(_agent_turns()),
            "--output-format",
            "json",
            # A mutation subprocess must not run the host's interactive session
            # hooks. They exist for a human's workflow, they can fail in a
            # headless run, and a failing teardown hook makes the session exit
            # nonzero after the edit is already on disk. The --bare flag also
            # skips hooks but forces API key auth, which breaks an OAuth host.
            "--settings",
            '{"hooks":{}}',
        ]
    return [
        program,
        "exec",
        "--skip-git-repo-check",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-o",
        str(workspace / "last-message.txt"),
        AGENT_TASK_PROMPT,
    ]


def _prepare_workspace(workspace: Path, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    resolved_root = workdir.resolve()
    resolved_workspace = workspace.resolve()
    if resolved_workspace.parent != resolved_root:
        raise OperatorError("agent workspace escaped the configured workdir")
    if workspace.exists():
        # A previous agent process can still hold a handle here on Windows,
        # where deleting an open directory raises PermissionError. That is not
        # a mutation failure, so it must never propagate and kill the worker.
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)


def _safe_path(workspace: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise OperatorError(f"unsafe parent file path: {relative_path}")
    target = workspace / relative
    try:
        target.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise OperatorError(f"unsafe parent file path: {relative_path}") from exc
    return target


def _agent_contract(bundle: ParentBundle, ctx: OperatorContext) -> str:
    contract = ctx.contract
    direction = "maximize" if contract.maximize else "minimize"
    target = "no fixed target, push as far as possible" if contract.target is None else str(
        contract.target
    )
    inspiration_scores = [
        f"- {program.id} ({program.code_ref}): "
        + ", ".join(f"{key}={value}" for key, value in sorted(scores.items()))
        for program, scores in bundle.inspirations[:3]
    ]
    discoveries = [f"- {item}" for item in bundle.discoveries]
    return "\n".join(
        (
            "# Mutation contract",
            "",
            f"Goal: {contract.goal}",
            f"Metric: {contract.metric} ({direction})",
            f"Target: {target}",
            f"Baseline: {contract.baseline if contract.baseline is not None else 'unmeasured'}",
            f"Gate: {contract.gate}",
            f"Parent: {bundle.parent.id}",
            "Parent scores: "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(bundle.parent_scores.items()))
                or "not recorded"
            ),
            "Best scores in this run: "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(bundle.best_scores.items()))
                or "not recorded"
            ),
            "Other elites in the population and their scores:",
            *(inspiration_scores or ["- None available yet."]),
            "Recent gate failures in this run, do not repeat them:",
            *([f"- {reason}" for reason in bundle.recent_failures] or ["- None recorded."]),
            "Prior discoveries:",
            *(discoveries or ["- None supplied."]),
            "",
            "You may run code. Write a scratch script somewhere outside this "
            "directory, execute it, and read the result. On a search problem that "
            "is the point: prototype a construction, measure it, keep what wins. "
            "Do not guess a recalled answer when you can compute a better one.",
            "",
            "You may edit files in place. Change ONLY content between lines containing "
            "EVOLVE-BLOCK-START and EVOLVE-BLOCK-END. Preserve marker lines and every "
            "byte outside those regions. Do not add or remove parent files. Leave a "
            "scratch file behind only outside this directory. Finish after making "
            "the strongest contract-respecting improvement you can, and finish "
            "having actually changed something.",
        )
    )
