"""Held-out Nguyen-7 rediscovery evaluator with an AST complexity penalty."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="train-gate", timeout_s=15.0),
    StageSpec(name="heldout-fitness", timeout_s=30.0),
]
GATE: str = "finite"

PACK_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = PACK_DIR / "fixtures"
Row = tuple[float, float]


def _load_module(candidate_dir: Path) -> ModuleType:
    entry_path = candidate_dir / "model.py"
    if not entry_path.is_file():
        raise EvalError(f"candidate is missing {entry_path.name}")
    module_name = f"_autoevolve_symbolic_regression_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_rows(name: str) -> list[Row]:
    raw = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return [(float(item["x"]), float(item["y"])) for item in raw["data"]]


def _predict(module: ModuleType, rows: list[Row], split: str) -> list[float]:
    predictions: list[float] = []
    for index, (x, _) in enumerate(rows):
        try:
            raw_value = module.predict(x)
        except Exception as exc:
            raise EvalError(f"{split} row {index} raised {type(exc).__name__}: {exc}") from exc
        if isinstance(raw_value, bool) or not isinstance(raw_value, float):
            raise EvalError(f"{split} row {index} returned a non-float prediction")
        if not math.isfinite(raw_value):
            raise EvalError(f"{split} row {index} returned a non-finite prediction")
        predictions.append(raw_value)
    return predictions


def _complexity(candidate_dir: Path) -> int:
    entry_path = candidate_dir / "model.py"
    tree = ast.parse(entry_path.read_text(encoding="utf-8"), filename=str(entry_path))
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
    """Gate finite train predictions, then score held-out fit and complexity."""
    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    candidate = _load_module(candidate_dir)
    train = _load_rows("train.json")
    _predict(candidate, train, "train")
    complexity = _complexity(candidate_dir)
    if stage == 0:
        return {GATE: 1.0, "complexity": float(complexity)}
    heldout = _load_rows("heldout.json")
    heldout_predictions = _predict(candidate, heldout, "heldout")
    r2_heldout = _r2(heldout, heldout_predictions)
    fitness = r2_heldout - 0.001 * complexity
    return {
        GATE: 1.0,
        "r2_heldout": r2_heldout,
        "complexity": float(complexity),
        "fitness": fitness,
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no fixed ceiling because the AST penalty depends on the candidate."""
    return None
