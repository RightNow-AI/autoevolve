"""Evaluator author imports and the engine-side evaluator metadata loader."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.sandbox import _kill_process_tree, _start_isolated_process

_RUNNER_TIMEOUT_S = 30.0
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


@dataclass(frozen=True)
class Evaluator:
    """Validated evaluator metadata that is safe to hold in the engine process.

    metric and maximize are the evaluator's own primary-metric declaration.
    When present, the engine locks the contract to them instead of guessing.
    """

    dir: Path
    stages: list[StageSpec]
    gate: str
    has_ceiling: bool
    spec_text: str
    metric: str | None = None
    maximize: bool | None = None
    descriptors: list[dict[str, Any]] | None = None

    def ceiling(self) -> dict[str, Any] | None:
        """Load the optional ceiling in an isolated runner process."""

        payload = _invoke_runner("--ceiling", self.dir)
        ceiling = payload.get("ceiling")
        if ceiling is not None and not isinstance(ceiling, dict):
            raise EvalError("evaluator ceiling() must return a dict or None")
        return ceiling


def _runner_env() -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name in _ALLOWED_ENV}
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return env


def _stderr_suffix(stderr: str) -> str:
    tail = stderr[-500:].strip()
    return f"; stderr: {tail}" if tail else ""


def _invoke_runner(mode: str, evaluator_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        # -P keeps the working directory off sys.path. The child runs with the
        # candidate copy as its cwd, so without this the candidate directory is
        # sys.path[0] and any module the runner or evaluator imports can be
        # shadowed by a file the candidate ships, running candidate code inside
        # the judging process. Python 3.11+.
        "-P",
        "-m",
        "autoevolve.eval.runner",
        mode,
        str(evaluator_dir),
    ]
    try:
        process = _start_isolated_process(
            command,
            evaluator_dir,
            _runner_env(),
        )
    except OSError as exc:
        raise EvalError(f"could not start evaluator runner: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=_RUNNER_TIMEOUT_S)
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
        action = mode.removeprefix("--")
        raise EvalError(f"evaluator {action} timed out after 30s") from exc

    lines = stdout.splitlines()
    if not lines:
        raise EvalError(
            "evaluator runner produced no JSON response" + _stderr_suffix(stderr)
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise EvalError(
            "evaluator runner returned invalid JSON" + _stderr_suffix(stderr)
        ) from exc
    if not isinstance(payload, dict):
        raise EvalError("evaluator runner response must be a JSON object")
    if payload.get("ok") is not True:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = "evaluator runner failed without a reason"
        raise EvalError(reason)
    return payload


def _positive_number(value: object, field: str, stage_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvalError(f"stage {stage_index} {field} must be a positive number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise EvalError(f"stage {stage_index} {field} must be a positive number") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise EvalError(f"stage {stage_index} {field} must be a positive number")
    return numeric


def _stage_from_json(value: object, stage_index: int) -> StageSpec:
    if not isinstance(value, dict):
        raise EvalError(f"stage {stage_index} description must be a JSON object")

    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise EvalError(f"stage {stage_index} name must be a non-empty string")
    timeout_s = _positive_number(value.get("timeout_s"), "timeout_s", stage_index)

    mem_value = value.get("mem_mb")
    if mem_value is not None and (
        isinstance(mem_value, bool) or not isinstance(mem_value, int) or mem_value <= 0
    ):
        raise EvalError(f"stage {stage_index} mem_mb must be a positive integer or None")

    cpu_value = value.get("cpu_s")
    cpu_s = None
    if cpu_value is not None:
        cpu_s = _positive_number(cpu_value, "cpu_s", stage_index)

    return StageSpec(name=name, timeout_s=timeout_s, mem_mb=mem_value, cpu_s=cpu_s)


def load_evaluator(evaluator_dir: Path) -> Evaluator:
    """Validate an evaluator folder and load only its child-reported metadata."""

    directory = Path(evaluator_dir).resolve()
    if not directory.is_dir():
        raise EvalError(f"evaluator directory does not exist: {directory}")
    if not (directory / "evaluate.py").is_file():
        raise EvalError(f"evaluator is missing evaluate.py: {directory}")
    if not (directory / "baseline").is_dir():
        raise EvalError(f"evaluator is missing baseline/: {directory}")

    spec_path = directory / "spec.md"
    if spec_path.exists() and not spec_path.is_file():
        raise EvalError(f"evaluator spec.md is not a file: {spec_path}")
    try:
        spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    except (OSError, UnicodeError) as exc:
        raise EvalError(f"could not read evaluator spec.md: {exc}") from exc

    payload = _invoke_runner("--describe", directory)
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise EvalError("evaluator STAGES must be a non-empty list")
    stages = [_stage_from_json(stage, index) for index, stage in enumerate(raw_stages)]

    gate = payload.get("gate")
    if not isinstance(gate, str) or not gate:
        raise EvalError("evaluator GATE must be a non-empty string")
    has_ceiling = payload.get("has_ceiling")
    if not isinstance(has_ceiling, bool):
        raise EvalError("evaluator description has invalid has_ceiling metadata")
    metric = payload.get("metric")
    if metric is not None and (not isinstance(metric, str) or not metric):
        raise EvalError("evaluator METRIC must be a non-empty string when declared")
    maximize = payload.get("maximize")
    if maximize is not None and not isinstance(maximize, bool):
        raise EvalError("evaluator MAXIMIZE must be a bool when declared")
    raw_descriptors = payload.get("descriptors")
    if raw_descriptors is not None and not isinstance(raw_descriptors, list):
        raise EvalError("evaluator DESCRIPTORS must be a list when declared")
    descriptors = list(raw_descriptors) if raw_descriptors else None

    return Evaluator(
        dir=directory,
        stages=stages,
        gate=gate,
        has_ceiling=has_ceiling,
        spec_text=spec_text,
        metric=metric,
        maximize=maximize,
        descriptors=descriptors,
    )


__all__ = ["EvalError", "Evaluator", "StageSpec", "load_evaluator"]
