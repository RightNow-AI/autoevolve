"""Deterministic toy evaluator used by the U2 integration tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from autoevolve.eval.contract import EvalError, StageSpec

STAGES = [
    StageSpec(name="smoke", timeout_s=20.0),
    StageSpec(name="full", timeout_s=30.0),
]
GATE = "correct"


def _load_solution(candidate_dir: Path) -> Callable[[list[int]], list[int]]:
    solution_path = candidate_dir / "solution.py"
    spec = importlib.util.spec_from_file_location("_autoevolve_toy_solution", solution_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"could not load candidate solution.py from {candidate_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise EvalError("candidate solution.py must define callable solve(xs)")
    return cast(Callable[[list[int]], list[int]], solve)


def _load_cases() -> list[dict[str, list[int]]]:
    raw = json.loads((Path(__file__).parent / "fixtures" / "cases.json").read_text("utf-8"))
    if not isinstance(raw, list):
        raise EvalError("toy evaluator cases.json must contain a list")
    return cast(list[dict[str, list[int]]], raw)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Check sorting parity and return a deterministic source compactness score."""

    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown evaluator stage {stage}")
    solve = _load_solution(candidate_dir)
    cases = _load_cases()
    selected_cases = cases[:3] if stage == 0 else cases
    for index, case in enumerate(selected_cases):
        actual = solve(list(case["input"]))
        if actual != case["expected"]:
            raise EvalError(f"case {index} failed")

    source = (candidate_dir / "solution.py").read_text(encoding="utf-8")
    return {"correct": 1.0, "score": 1000.0 / max(len(source), 1)}
