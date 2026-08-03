"""Engine-side subprocess sandbox for one evaluator stage."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoevolve.core.types import EvalError, StageSpec

if sys.platform != "win32":
    import resource

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_ENV = frozenset(
    {
        "HOME",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
    }
)


_ENGINE_ONLY_ENV = frozenset(
    {
        "AUTOEVOLVE_HOME",
        "AUTOEVOLVE_ARTIFACTS_DIR",
        "AUTOEVOLVE_LOCAL_BASE_URL",
        "AUTOEVOLVE_LOCAL_MODEL",
        "AUTOEVOLVE_AGENT_RUNTIME",
        "AUTOEVOLVE_AGENTIC_TIMEOUT_S",
    }
)


def _sandbox_env() -> dict[str, str]:
    """Build the scrubbed child environment.

    The allowlist keeps secrets out of candidate reach. AUTOEVOLVE_ prefixed
    variables are configuration this project sets deliberately, and cells
    select their workload through them, so a campaign cannot express a
    multi-cell campaign at all without them. Credential shaped names are
    excluded even under that prefix, so configuration can never smuggle a
    key into a candidate's environment.
    """

    env = {name: value for name, value in os.environ.items() if name in _ALLOWED_ENV}
    for name, value in os.environ.items():
        if name.startswith("AUTOEVOLVE_") and _is_candidate_visible(name):
            env[name] = value
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return env


def _is_candidate_visible(name: str) -> bool:
    """Workload configuration is visible; engine and model configuration is not.

    AUTOEVOLVE_HOME points at the run database, so a candidate holding it
    could edit its own scores directly. Model and endpoint settings are
    likewise none of a candidate's business, and credential shaped names
    never pass under any prefix.
    """

    if any(
        token in name for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    ):
        return False
    return name not in _ENGINE_ONLY_ENV and not name.startswith("AUTOEVOLVE_MODEL")


if sys.platform != "win32":

    def _posix_limits(spec: StageSpec) -> Callable[[], None]:
        def apply_limits() -> None:
            if spec.mem_mb is not None:
                mem_bytes = spec.mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            if spec.cpu_s is not None:
                cpu_seconds = max(1, math.ceil(spec.cpu_s))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        return apply_limits


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            process.kill()


def _start_isolated_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    preexec_fn: Callable[[], None] | None = None,
) -> subprocess.Popen[str]:
    common: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            **common,
        )
    return subprocess.Popen(
        command,
        start_new_session=True,
        preexec_fn=preexec_fn,
        **common,
    )


def _start_runner(
    command: list[str],
    cwd: Path,
    spec: StageSpec,
) -> subprocess.Popen[str]:
    # Retain the child PID so timeout handling can terminate the complete process tree.
    if sys.platform == "win32":
        return _start_isolated_process(command, cwd, _sandbox_env())
    return _start_isolated_process(
        command,
        cwd,
        _sandbox_env(),
        preexec_fn=_posix_limits(spec),
    )


def _parse_response(stdout: str, stderr: str, returncode: int | None = None) -> dict[str, float]:
    """Read the runner's verdict, taking the FIRST one and never a later line.

    The runner emits exactly one line and then leaves through os._exit, so a
    second verdict line can only come from something trying to overwrite the
    first. Taking the last line would hand the decision to whoever spoke
    last, which is precisely the forgery this ordering prevents.
    """

    if returncode is not None and returncode != 0:
        raise EvalError(
            f"evaluator runner exited with code {returncode}; stderr: {stderr[-500:]}"
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise EvalError(f"invalid evaluator response; stderr: {stderr[-500:]}")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid evaluator response; stderr: {stderr[-500:]}") from exc
    if not isinstance(payload, dict):
        raise EvalError(f"invalid evaluator response; stderr: {stderr[-500:]}")
    if payload.get("ok") is not True:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = "evaluator failed without a reason"
        raise EvalError(reason)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise EvalError("evaluator success response is missing a metrics object")
    return metrics


def run_stage(
    evaluator_dir: Path,
    candidate_dir: Path,
    stage: int,
    spec: StageSpec,
) -> dict[str, float]:
    """Run one evaluator stage against a fresh temporary candidate copy."""

    evaluator_path = Path(evaluator_dir).resolve()
    candidate_path = Path(candidate_dir).resolve()
    if not evaluator_path.is_dir():
        raise EvalError(f"evaluator directory does not exist: {evaluator_path}")
    if not candidate_path.is_dir():
        raise EvalError(f"candidate directory does not exist: {candidate_path}")

    with tempfile.TemporaryDirectory(prefix="autoevolve-eval-") as temp_raw:
        temp_dir = Path(temp_raw)
        candidate_copy = temp_dir / "candidate"
        try:
            shutil.copytree(candidate_path, candidate_copy, dirs_exist_ok=True)
        except OSError as exc:
            raise EvalError(f"could not prepare evaluator sandbox: {exc}") from exc

        command = [
            sys.executable,
            "-m",
            "autoevolve.eval.runner",
            "--evaluate",
            str(evaluator_path),
            str(candidate_copy),
            str(stage),
        ]

        try:
            process = _start_runner(command, candidate_copy, spec)
        except OSError as exc:
            raise EvalError(f"could not start evaluator sandbox: {exc}") from exc
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    process.kill()
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            raise EvalError(
                f"timeout after {spec.timeout_s}s at stage {spec.name}"
            ) from exc

    return _parse_response(stdout, stderr, process.returncode)
