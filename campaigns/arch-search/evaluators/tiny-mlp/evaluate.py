"""Fixed-budget NumPy training evaluator for a small MLP."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [StageSpec(name="fixed-training", timeout_s=30.0)]
GATE = "trained"
PACK_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PACK_DIR / "fixtures" / "data.json"
EPOCHS = 32
BATCH_SIZE = 16
LEARNING_RATE = 0.08
TRAINING_SEED = 2_718


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "model.py"
    if not path.is_file():
        raise EvalError("candidate is missing model.py")
    spec = importlib.util.spec_from_file_location(
        f"_autoevolve_tiny_mlp_{abs(hash(path.resolve()))}",
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


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray([[row["x1"], row["x2"]] for row in rows], dtype=float)
        labels = np.asarray([[row["label"]] for row in rows], dtype=float)
        return features, labels

    train_x, train_y = arrays(payload["train"])
    validation_x, validation_y = arrays(payload["validation"])
    return train_x, train_y, validation_x, validation_y


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _loss(predictions: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(predictions, 1e-9, 1.0 - 1e-9)
    value = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(value)


def _train(module: ModuleType) -> tuple[float, float, int]:
    train_x, train_y, validation_x, validation_y = _load_data()
    try:
        hidden = int(module.HIDDEN_DIM)
        init_scale = float(module.INIT_SCALE)
        activate = module.activation
        activate_grad = module.activation_grad
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvalError(f"candidate interface is invalid: {exc}") from exc
    if hidden <= 0 or not math.isfinite(init_scale) or init_scale <= 0.0:
        raise EvalError("candidate hidden size and initialization scale must be positive")

    rng = np.random.default_rng(TRAINING_SEED)
    w1 = rng.normal(0.0, init_scale, size=(2, hidden))
    b1 = np.zeros((1, hidden), dtype=float)
    w2 = rng.normal(0.0, init_scale, size=(hidden, 1))
    b2 = np.zeros((1, 1), dtype=float)

    def forward(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = features @ w1 + b1
        h1 = np.asarray(activate(z1), dtype=float)
        predictions = _sigmoid(h1 @ w2 + b2)
        return z1, h1, predictions

    _, _, initial_predictions = forward(train_x)
    first_loss = _loss(initial_predictions, train_y)
    for _ in range(EPOCHS):
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), BATCH_SIZE):
            indexes = order[start : start + BATCH_SIZE]
            batch_x = train_x[indexes]
            batch_y = train_y[indexes]
            z1, h1, predictions = forward(batch_x)
            output_delta = (predictions - batch_y) / len(batch_x)
            grad_w2 = h1.T @ output_delta
            grad_b2 = np.sum(output_delta, axis=0, keepdims=True)
            derivative = np.asarray(activate_grad(z1), dtype=float)
            hidden_delta = (output_delta @ w2.T) * derivative
            grad_w1 = batch_x.T @ hidden_delta
            grad_b1 = np.sum(hidden_delta, axis=0, keepdims=True)
            w1 -= LEARNING_RATE * grad_w1
            b1 -= LEARNING_RATE * grad_b1
            w2 -= LEARNING_RATE * grad_w2
            b2 -= LEARNING_RATE * grad_b2

    _, _, final_predictions = forward(train_x)
    _, _, validation_predictions = forward(validation_x)
    final_loss = _loss(final_predictions, train_y)
    validation_loss = _loss(validation_predictions, validation_y)
    if not all(math.isfinite(value) for value in (first_loss, final_loss, validation_loss)):
        raise EvalError("training produced a non-finite loss")
    if final_loss >= first_loss:
        raise EvalError(
            f"training loss did not decrease: first={first_loss}, final={final_loss}"
        )
    params = w1.size + b1.size + w2.size + b2.size
    return final_loss, validation_loss, int(params)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Train for the fixed budget and return validation loss plus parameter count."""

    if stage != 0:
        raise EvalError(f"unknown stage {stage}")
    if os.environ.get("AUTOEVOLVE_CELL", "tiny-mlp") != "tiny-mlp":
        raise EvalError("AUTOEVOLVE_CELL must be tiny-mlp for this evaluator")
    module = _load_candidate(candidate_dir)
    train_loss, validation_loss, params = _train(module)
    return {
        GATE: 1.0,
        "params": float(params),
        "train_loss": train_loss,
        "val_loss": validation_loss,
        **source_metrics(candidate_dir, "model.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no theoretical validation-loss ceiling."""

    return None


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
