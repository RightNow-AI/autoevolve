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


def _sandbox_env() -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name in _ALLOWED_ENV}
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return env


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


def _start_runner(
    command: list[str],
    cwd: Path,
    spec: StageSpec,
) -> subprocess.Popen[str]:
    # Retain the child PID so timeout handling can terminate the complete process tree.
    common: dict[str, Any] = {
        "cwd": cwd,
        "env": _sandbox_env(),
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
        preexec_fn=_posix_limits(spec),
        **common,
    )


def _parse_response(stdout: str, stderr: str) -> dict[str, float]:
    lines = stdout.splitlines()
    if not lines:
        raise EvalError(f"invalid evaluator response; stderr: {stderr[-500:]}")
    try:
        payload = json.loads(lines[-1])
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
                process.communicate()
            raise EvalError(
                f"timeout after {spec.timeout_s}s at stage {spec.name}"
            ) from exc

    return _parse_response(stdout, stderr)
