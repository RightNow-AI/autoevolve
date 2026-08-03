"""Child process entrypoint for evaluator discovery and execution."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, TextIO

from autoevolve.eval.contract import EvalError, StageSpec

_NETWORK_DISABLED_REASON = "network disabled in sandbox"


class _BlockedSocket:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EvalError(_NETWORK_DISABLED_REASON)


def _blocked_create_connection(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise EvalError(_NETWORK_DISABLED_REASON)


def _install_socket_block() -> None:
    socket.socket = _BlockedSocket  # type: ignore[assignment]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]


def _load_module(evaluator_dir: Path) -> ModuleType:
    evaluate_path = evaluator_dir / "evaluate.py"
    module_name = f"_autoevolve_evaluator_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, evaluate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load evaluator module from {evaluate_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    evaluator_path = str(evaluator_dir)
    sys.path.insert(0, evaluator_path)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(evaluator_path)
        except ValueError:
            pass
    return module


def _describe(evaluator_dir: Path) -> dict[str, Any]:
    module = _load_module(evaluator_dir)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise TypeError("evaluate.py must define callable evaluate()")

    stages = getattr(module, "STAGES", None)
    if not isinstance(stages, list) or not stages:
        raise TypeError("evaluate.py STAGES must be a non-empty list")
    stage_payloads: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, StageSpec):
            raise TypeError(f"STAGES[{index}] must be a StageSpec")
        stage_payloads.append(
            {
                "name": stage.name,
                "timeout_s": stage.timeout_s,
                "mem_mb": stage.mem_mb,
                "cpu_s": stage.cpu_s,
            }
        )

    gate = getattr(module, "GATE", None)
    if not isinstance(gate, str) or not gate:
        raise TypeError("evaluate.py GATE must be a non-empty string")
    ceiling = getattr(module, "ceiling", None)
    if ceiling is not None and not callable(ceiling):
        raise TypeError("evaluate.py ceiling must be callable when defined")
    metric = getattr(module, "METRIC", None)
    if metric is not None and (not isinstance(metric, str) or not metric):
        raise TypeError("evaluate.py METRIC must be a non-empty string when defined")
    maximize = getattr(module, "MAXIMIZE", None)
    if maximize is not None and not isinstance(maximize, bool):
        raise TypeError("evaluate.py MAXIMIZE must be a bool when defined")
    return {
        "stages": stage_payloads,
        "gate": gate,
        "has_ceiling": callable(ceiling),
        "metric": metric,
        "maximize": maximize,
    }


def _ceiling(evaluator_dir: Path) -> dict[str, Any] | None:
    module = _load_module(evaluator_dir)
    ceiling = getattr(module, "ceiling", None)
    if ceiling is None:
        return None
    if not callable(ceiling):
        raise TypeError("evaluate.py ceiling must be callable when defined")
    result = ceiling()
    if result is not None and not isinstance(result, dict):
        raise TypeError("ceiling() must return a dict or None")
    return result


def _evaluate(evaluator_dir: Path, candidate_dir: Path, stage: int) -> dict[str, Any]:
    module = _load_module(evaluator_dir)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise TypeError("evaluate.py must define callable evaluate()")
    metrics = evaluate(candidate_dir, stage)
    if not isinstance(metrics, dict):
        raise TypeError("evaluate() must return a metrics dict")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", metavar="EVALUATOR_DIR", type=Path)
    modes.add_argument("--ceiling", metavar="EVALUATOR_DIR", type=Path)
    modes.add_argument(
        "--evaluate",
        nargs=3,
        metavar=("EVALUATOR_DIR", "CANDIDATE_DIR", "STAGE"),
    )
    return parser


def _crash_reason(exc: Exception) -> str:
    return f"crash: {type(exc).__name__}: {exc}"


def _emit(stream: TextIO, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    print(encoded, file=stream, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.describe is not None:
                payload = {"ok": True, **_describe(args.describe.resolve())}
            elif args.ceiling is not None:
                payload = {"ok": True, "ceiling": _ceiling(args.ceiling.resolve())}
            else:
                evaluator_raw, candidate_raw, stage_raw = args.evaluate
                _install_socket_block()
                metrics = _evaluate(
                    Path(evaluator_raw).resolve(),
                    Path(candidate_raw).resolve(),
                    int(stage_raw),
                )
                payload = {"ok": True, "metrics": metrics}
    except EvalError as exc:
        payload = {"ok": False, "reason": exc.reason}
    except Exception as exc:
        payload = {"ok": False, "reason": _crash_reason(exc)}

    try:
        _emit(real_stdout, payload)
    except Exception as exc:
        _emit(real_stdout, {"ok": False, "reason": _crash_reason(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
