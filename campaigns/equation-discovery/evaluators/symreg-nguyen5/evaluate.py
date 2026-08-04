"""Held-out Nguyen-5 rediscovery evaluator with a complexity penalty."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [StageSpec(name="heldout-fitness", timeout_s=30.0)]
GATE = "finite"
PACK_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = PACK_DIR / "fixtures"
Row = tuple[float, float]


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "model.py"
    if not path.is_file():
        raise EvalError("candidate is missing model.py")
    spec = importlib.util.spec_from_file_location(
        f"_autoevolve_nguyen5_{abs(hash(path.resolve()))}",
        path,
    )
    if spec is None or spec.loader is None:
        raise EvalError("candidate model.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_rows(name: str) -> list[Row]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return [(float(row["x"]), float(row["y"])) for row in payload["data"]]


def _predict(module: ModuleType, rows: list[Row], split: str) -> list[float]:
    predictions: list[float] = []
    for index, (x, _) in enumerate(rows):
        try:
            value = module.predict(x)
        except Exception as exc:
            raise EvalError(f"{split} row {index} failed: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, float):
            raise EvalError(f"{split} row {index} returned a non-float prediction")
        if not math.isfinite(value):
            raise EvalError(f"{split} row {index} returned a non-finite prediction")
        predictions.append(value)
    return predictions


def _complexity(candidate_dir: Path) -> int:
    path = candidate_dir / "model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "predict":
            return sum(1 for statement in node.body for _ in ast.walk(statement))
    raise EvalError("candidate does not define predict")


def _r2(rows: list[Row], predictions: list[float]) -> float:
    targets = [target for _, target in rows]
    mean_target = sum(targets) / len(targets)
    residual = sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    )
    total = sum((target - mean_target) ** 2 for target in targets)
    return 1.0 - residual / total


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate finite predictions and score held-out fit plus complexity."""

    if stage != 0:
        raise EvalError(f"unknown stage {stage}")
    if os.environ.get("AUTOEVOLVE_CELL", "nguyen-5") != "nguyen-5":
        raise EvalError("AUTOEVOLVE_CELL must be nguyen-5 for this evaluator")
    candidate = _load_candidate(candidate_dir)
    train = _load_rows("train.json")
    heldout = _load_rows("heldout.json")
    _predict(candidate, train, "train")
    heldout_predictions = _predict(candidate, heldout, "heldout")
    complexity = _complexity(candidate_dir)
    r2_heldout = _r2(heldout, heldout_predictions)
    fitness = r2_heldout - 0.001 * complexity
    return {
        GATE: 1.0,
        "complexity": float(complexity),
        "fitness": fitness,
        "r2_heldout": r2_heldout,
        **source_metrics(candidate_dir, "model.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no fixed ceiling because complexity depends on the candidate."""

    return None


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
