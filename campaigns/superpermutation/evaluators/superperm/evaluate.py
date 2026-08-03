"""Superpermutation evaluator with isolated construction and an exact total gate."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="build-and-verify", timeout_s=90.0),
    StageSpec(name="determinism-replay", timeout_s=240.0),
]
GATE = "complete"
METRIC = "length"
MAXIMIZE = False

DESCRIPTORS = [
    {
        "name": "max_perm_gap",
        "metric": "max_perm_gap",
        "bins": 8,
        "lo": 1.0,
        "hi": 32.0,
    },
    {
        "name": "perm_gap_kinds",
        "metric": "perm_gap_kinds",
        "bins": 8,
        "lo": 1.0,
        "hi": 16.0,
    },
]

_len, _set, _frozenset, _type, _range, _int = len, set, frozenset, type, range, int
_CELL_RE = re.compile(r"\An([2-7])\Z")
_CELL_RAW = os.environ.get("AUTOEVOLVE_CELL")
_MATCH = _CELL_RE.match(_CELL_RAW) if _type(_CELL_RAW) is str else None
_N = _int(_MATCH.group(1)) if _MATCH is not None else None
_EVAL_DIR = Path(__file__).resolve().parent

_MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
_MAX_CAPTURE_BYTES = 1_000_001


def _load_verifier() -> ModuleType:
    path = _EVAL_DIR / "verify_cert.py"
    spec = importlib.util.spec_from_file_location("_superperm_verify", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load verify_cert.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_VERIFY = _load_verifier()


def _candidate_size(candidate_dir: Path) -> int:
    total = 0
    try:
        for path in candidate_dir.rglob("*"):
            if path.is_symlink():
                raise EvalError(f"candidate contains a symbolic link: {path.name}")
            if not path.is_file():
                continue
            total += path.stat().st_size
            if total > _MAX_CANDIDATE_BYTES:
                raise EvalError("candidate directory exceeds the 64 MiB size limit")
    except OSError as exc:
        raise EvalError(f"could not inspect candidate directory: {exc}") from exc
    return total


def _child_environment(work_dir: Path, deadline: float) -> dict[str, str]:
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "TEMP": str(work_dir / "tmp"),
        "TMP": str(work_dir / "tmp"),
        "AUTOEVOLVE_BUILD_DEADLINE": repr(deadline),
    }
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _start_process(
    command: list[str],
    work_dir: Path,
    env: dict[str, str],
    stdout: Any,
    stderr: Any,
) -> subprocess.Popen[bytes]:
    common: dict[str, Any] = {
        "cwd": work_dir,
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
    }
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            **common,
        )
    return subprocess.Popen(command, start_new_session=True, **common)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
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


def _stderr_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            return handle.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _build_once(candidate_dir: Path, n: int, budget_s: float) -> bytes:
    with tempfile.TemporaryDirectory(prefix="autoevolve-superperm-") as raw:
        work_dir = Path(raw)
        candidate_copy = work_dir / "candidate"
        try:
            shutil.copytree(
                candidate_dir,
                candidate_copy,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copyfile(_EVAL_DIR / "build_runner.py", work_dir / "build_runner.py")
            (work_dir / "tmp").mkdir()
        except OSError as exc:
            raise EvalError(f"could not prepare isolated build: {exc}") from exc

        certificate_path = work_dir / "certificate.bin"
        stderr_path = work_dir / "stderr.log"
        command = [
            sys.executable,
            "-P",
            "-s",
            "-B",
            str(work_dir / "build_runner.py"),
            str(candidate_copy),
            str(n),
        ]
        deadline = time.monotonic() + budget_s
        env = _child_environment(work_dir, deadline)
        try:
            with certificate_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = _start_process(command, work_dir, env, stdout, stderr)
                try:
                    process.wait(timeout=budget_s)
                except subprocess.TimeoutExpired as exc:
                    _kill_process_tree(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if process.poll() is None:
                            process.kill()
                    raise EvalError(
                        f"candidate build timed out after {budget_s:.3f} seconds"
                    ) from exc
        except OSError as exc:
            raise EvalError(f"could not run isolated build: {exc}") from exc

        if process.returncode != 0:
            detail = _stderr_tail(stderr_path)
            suffix = f": {detail}" if detail else ""
            raise EvalError(f"candidate build failed with code {process.returncode}{suffix}")
        try:
            with certificate_path.open("rb") as handle:
                return handle.read(_MAX_CAPTURE_BYTES)
        except OSError as exc:
            raise EvalError(f"could not read candidate certificate: {exc}") from exc


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Build an isolated certificate, gate it exactly, and minimize its byte length."""

    if _N is None:
        raise EvalError(
            "AUTOEVOLVE_CELL is unset or malformed; expected n2..n7 "
            f"(saw {_CELL_RAW!r})"
        )
    if stage not in _range(_len(STAGES)):
        raise EvalError(f"unknown stage {stage}")
    root = Path(candidate_dir).resolve()
    if not root.is_dir():
        raise EvalError(f"candidate directory does not exist: {root}")
    if not (root / "builder.py").is_file():
        raise EvalError("candidate is missing builder.py")
    _candidate_size(root)

    if stage == 0:
        certificate = _build_once(root, _N, 72.0)
    else:
        first = _build_once(root, _N, 108.0)
        second = _build_once(root, _N, 108.0)
        if first != second:
            raise EvalError("nondeterministic certificate: two fresh builds disagree")
        certificate = first
    verify = _VERIFY.verify_certificate
    return verify(certificate, _N)


def ceiling() -> dict[str, float | str] | None:
    """There is no evaluator-computed lower ceiling for this minimization problem."""

    return None
