"""Run the autoevolve agentic operator entirely on Modal.

Usage:
    modal run scripts/modal_agentic.py::preflight_local
    modal run scripts/modal_agentic.py::agentic_search_local \
        --evaluator campaigns/example/evaluator --goal "Improve the candidate"
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"


def _head_sha() -> str:
    """Return the repository revision that the Modal image must contain."""

    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
    )
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} /root/autoevolve",
        f"cd /root/autoevolve && git checkout {COMMIT}",
        "cd /root/autoevolve && uv sync --frozen",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-agentic")


def _configure_codex() -> dict[str, str]:
    """Prepare non-interactive Codex authentication and provider configuration."""

    import json
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Modal secret autoevolve-model did not supply OPENAI_API_KEY")

    # Verified 2026-08-05 against the current Codex documentation:
    # https://learn.chatgpt.com/docs/config-file/environment-variables#authentication-and-network
    # documents CODEX_API_KEY for one non-interactive codex exec invocation.
    # https://learn.chatgpt.com/docs/config-file/config-reference#configtoml
    # documents openai_base_url for the built-in OpenAI provider. A custom
    # model_provider block is only needed when selecting a separate provider id.
    env = dict(os.environ)
    env["CODEX_API_KEY"] = api_key
    lines = [
        "# Generated inside the Modal container. No credential is stored here.",
        'approval_policy = "never"',
    ]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        lines.append(f"openai_base_url = {json.dumps(base_url)}")

    codex_home = Path.home() / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=900,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def preflight() -> dict[str, object]:
    """Prove that the installed Codex CLI makes a real workspace edit."""

    import json
    import subprocess
    import tempfile

    try:
        env = _configure_codex()
    except RuntimeError as exc:
        return {
            "ok": False,
            "changed": False,
            "expected_change": False,
            "exit_code": None,
            "runtime": "codex",
            "error": str(exc)[:800],
        }
    env["AUTOEVOLVE_AGENT_RUNTIME"] = "codex"

    with tempfile.TemporaryDirectory(prefix="autoevolve-codex-preflight-") as raw_workspace:
        workspace = Path(raw_workspace)
        target = workspace / "target.txt"
        original = "before\n"
        target.write_text(original, encoding="utf-8")
        (workspace / "PROMPT.md").write_text(
            "# Preflight contract\n\n"
            "Edit target.txt so its complete contents are the word `after` followed "
            "by one newline. "
            "Do not create or delete any files. Finish immediately after the edit.\n",
            encoding="utf-8",
        )

        # These flags come from the operator itself, not a second command copy.
        # The current CLI reference documents codex exec, -C, --sandbox, and -o:
        # https://learn.chatgpt.com/docs/developer-commands#codex-exec
        # The non-interactive guide documents --skip-git-repo-check:
        # https://learn.chatgpt.com/docs/non-interactive-mode#git-repository-required
        driver = """
import json
import subprocess
import sys
from pathlib import Path

from autoevolve.mutate.agentic import _agent_command, _failure_detail, _select_runtime
from autoevolve.mutate.base import OperatorError

workspace = Path(sys.argv[1])
try:
    runtime, executable = _select_runtime()
    command = _agent_command(runtime, workspace, executable)
    completed = subprocess.run(
        command,
        cwd=workspace,
        timeout=300,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    error = "" if completed.returncode == 0 else _failure_detail(completed)
    result = {
        "runtime": runtime,
        "exit_code": completed.returncode,
        "error": error,
    }
except subprocess.TimeoutExpired as exc:
    result = {
        "runtime": "codex",
        "exit_code": None,
        "error": f"codex timed out after {exc.timeout} seconds",
    }
except (OSError, OperatorError) as exc:
    result = {
        "runtime": "codex",
        "exit_code": None,
        "error": str(exc),
    }
print(json.dumps(result, sort_keys=True))
"""
        try:
            completed = subprocess.run(
                ["uv", "run", "python", "-c", driver, str(workspace)],
                cwd="/root/autoevolve",
                env=env,
                timeout=360,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            actual = target.read_text(encoding="utf-8") if target.is_file() else None
            return {
                "ok": actual == "after\n",
                "changed": actual != original,
                "expected_change": actual == "after\n",
                "exit_code": None,
                "runtime": "codex",
                "error": f"preflight driver timed out after {exc.timeout} seconds"[:800],
            }
        except OSError as exc:
            return {
                "ok": False,
                "changed": False,
                "expected_change": False,
                "exit_code": None,
                "runtime": "codex",
                "error": str(exc)[:800],
            }

        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        changed = actual != original
        expected_change = actual == "after\n"
        lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
        try:
            detail = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            detail = {}
        exit_code = detail.get("exit_code")
        runtime = str(detail.get("runtime", "codex"))
        error = str(detail.get("error", ""))[:800]
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout or "preflight driver failed")[:800]
        elif not detail:
            error = "preflight driver returned no structured verdict"
        elif not expected_change and not error:
            error = "codex exited without making the required edit"
        return {
            "ok": expected_change,
            "changed": changed,
            "expected_change": expected_change,
            "exit_code": exit_code,
            "runtime": runtime,
            "error": error,
        }


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=8.0,
    memory=16384,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def agentic_search(
    evaluator: str,
    goal: str,
    cell: str | None = None,
    store_name: str = "default",
    budget: int = 2000,
    parallel: int = 8,
    hours: float = 6.0,
    target: float | None = None,
) -> dict[str, object]:
    """Run one bounded agentic-only search and persist its store and artifacts."""

    import subprocess
    import threading

    env = _configure_codex()
    env["AUTOEVOLVE_AGENT_RUNTIME"] = "codex"
    env["AUTOEVOLVE_HOME"] = f"/store/{store_name}/autoevolve"
    env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"/store/{store_name}/runs"
    env["AUTOEVOLVE_AGENTIC_TIMEOUT_S"] = "1800"
    if cell:
        env["AUTOEVOLVE_CELL"] = cell

    command = [
        "uv",
        "run",
        "autoevolve",
        "run",
        "--evaluator",
        evaluator,
        "--goal",
        goal,
        "--budget-evals",
        str(budget),
        "--wall-clock-s",
        str(int(hours * 3600)),
        "--workers",
        str(parallel),
        "--parallel",
        str(parallel),
        "--operators",
        "agentic",
        "--seed",
        "1",
    ]
    if target is not None:
        command += ["--target", str(target)]

    finished = threading.Event()

    def checkpoint() -> None:
        while not finished.wait(120.0):
            try:
                store.commit()
            except Exception as exc:  # noqa: BLE001 - a later commit still has value
                print(f"checkpoint failed: {exc}", flush=True)

    keeper = threading.Thread(target=checkpoint, daemon=True)
    keeper.start()
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            command,
            cwd="/root/autoevolve",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        finished.set()
        keeper.join(timeout=10)
        store.commit()

    if completed is None:
        raise RuntimeError("autoevolve process ended without a completion record")
    stdout_tail = (completed.stdout or "")[-4000:]
    stderr_tail = (completed.stderr or "")[-2000:]
    print("=== autoevolve stdout tail ===", flush=True)
    print(stdout_tail, flush=True)
    if completed.returncode != 0:
        print("=== stderr tail ===", flush=True)
        print(stderr_tail, flush=True)
    print(f"=== exit {completed.returncode} ===", flush=True)
    return {
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.local_entrypoint()
def preflight_local() -> None:
    """Launch the mandatory remote Codex edit preflight."""

    import json

    print(json.dumps(preflight.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def agentic_search_local(
    evaluator: str,
    goal: str,
    cell: str | None = None,
    store_name: str = "default",
    budget: int = 2000,
    parallel: int = 8,
    hours: float = 6.0,
    target: float | None = None,
) -> None:
    """Launch one remote agentic-only search."""

    import json

    result = agentic_search.remote(
        evaluator,
        goal,
        cell,
        store_name,
        budget,
        parallel,
        hours,
        target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
