"""Evaluator that reads its cell at import time, as frontier packs must."""

from __future__ import annotations

import os
from pathlib import Path

from autoevolve.eval.contract import EvalError, StageSpec

_CELLS = ("small", "large")
_CELL = os.environ.get("AUTOEVOLVE_CELL")
if _CELL not in _CELLS:
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {', '.join(_CELLS)}; got {_CELL}")

STAGES: list[StageSpec] = [StageSpec(name="only", timeout_s=20.0)]
GATE: str = "correct"
METRIC: str = "size"
MAXIMIZE: bool = True


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    return {GATE: 1.0, "size": 1.0 if _CELL == "small" else 2.0}
