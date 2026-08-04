"""Fixed-budget NumPy evaluator for candidate optimizer update rules."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import FunctionType, ModuleType
from typing import cast

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [
    StageSpec(name="single-seed-proxy", timeout_s=30.0),
    StageSpec(name="three-seed-replication", timeout_s=60.0),
]
GATE: str = "trained"
METRIC: str = "val_loss"
MAXIMIZE: bool = False

PACK_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PACK_DIR / "fixtures" / "data.json"
STEPS = 300
HIDDEN_DIM = 16
INITIALIZATION_SEEDS = (20_260_803, 20_260_809, 20_260_821)
PARAMETER_NAMES = ("w1", "b1", "w2", "b2")

Parameters = dict[str, np.ndarray]
OptimizerState = dict[str, np.ndarray | float]
InitState = Callable[[tuple[int, ...]], object]
UpdateRule = Callable[[np.ndarray, np.ndarray, OptimizerState, int], object]
PreparedRun = tuple[int, Parameters]

_ARRAY = np.array
_ALL = np.all
_CLIP = np.clip
_DEFAULT_RNG = np.random.default_rng
_ERRSTATE = np.errstate
_EXP = np.exp
_FLOAT64 = np.float64
_ISFINITE = np.isfinite
_LOGADDEXP = np.logaddexp
_MEAN = np.mean
_SUM = np.sum
_TANH = np.tanh
_ZEROS = np.zeros
_NDARRAY_TYPE = np.ndarray
_ISFINITE_FLOAT = math.isfinite
_FSUM = math.fsum


def _load_candidate(candidate_dir: Path) -> tuple[InitState, UpdateRule]:
    entry_path = candidate_dir / "rule.py"
    if not entry_path.is_file():
        raise EvalError("candidate is missing rule.py")
    spec = importlib.util.spec_from_file_location(
        "_autoevolve_optimizer_rule_candidate",
        entry_path,
    )
    if spec is None or spec.loader is None:
        raise EvalError("candidate rule.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return _candidate_functions(module)


def _candidate_functions(module: ModuleType) -> tuple[InitState, UpdateRule]:
    namespace = vars(module)
    raw_init_state = namespace.get("init_state")
    raw_update = namespace.get("update")
    if type(raw_init_state) is not FunctionType:
        raise EvalError("candidate init_state must be a plain Python function")
    if type(raw_update) is not FunctionType:
        raise EvalError("candidate update must be a plain Python function")
    return cast(InitState, raw_init_state), cast(UpdateRule, raw_update)


def _load_rows(
    payload: dict[str, object],
    split: str,
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    raw_rows = payload.get(split)
    if type(raw_rows) is not list or not raw_rows:
        raise EvalError(f"fixture split {split} must be a non-empty list")

    features: list[list[float]] = []
    labels: list[list[float]] = []
    sample_ids: set[int] = set()
    for index, raw_row in enumerate(raw_rows):
        if type(raw_row) is not dict:
            raise EvalError(f"fixture split {split} row {index} must be an object")
        sample_id = raw_row.get("id")
        label = raw_row.get("label")
        x1 = raw_row.get("x1")
        x2 = raw_row.get("x2")
        if type(sample_id) is not int or sample_id in sample_ids:
            raise EvalError(f"fixture split {split} row {index} has an invalid id")
        if type(label) is not int or label not in (0, 1):
            raise EvalError(f"fixture split {split} row {index} has an invalid label")
        if type(x1) is not float or type(x2) is not float:
            raise EvalError(f"fixture split {split} row {index} has invalid features")
        if not _ISFINITE_FLOAT(x1) or not _ISFINITE_FLOAT(x2):
            raise EvalError(f"fixture split {split} row {index} has non-finite features")
        sample_ids.add(sample_id)
        features.append([x1, x2])
        labels.append([float(label)])

    feature_array = _ARRAY(features, dtype=_FLOAT64)
    label_array = _ARRAY(labels, dtype=_FLOAT64)
    feature_array.setflags(write=False)
    label_array.setflags(write=False)
    return feature_array, label_array, sample_ids


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        raw_payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvalError(f"could not load optimizer fixture: {exc}") from exc
    if type(raw_payload) is not dict:
        raise EvalError("optimizer fixture root must be an object")
    payload = cast(dict[str, object], raw_payload)
    train_x, train_y, train_ids = _load_rows(payload, "train")
    validation_x, validation_y, validation_ids = _load_rows(payload, "validation")
    overlap = train_ids & validation_ids
    if overlap:
        raise EvalError(f"training and validation fixtures overlap at sample id {min(overlap)}")
    return train_x, train_y, validation_x, validation_y


def _initial_parameters(seed: int) -> Parameters:
    rng = _DEFAULT_RNG(seed)
    parameters = {
        "w1": rng.normal(0.0, 0.25, size=(2, HIDDEN_DIM)).astype(_FLOAT64),
        "b1": _ZEROS((HIDDEN_DIM,), dtype=_FLOAT64),
        "w2": rng.normal(0.0, 0.25, size=(HIDDEN_DIM, 1)).astype(_FLOAT64),
        "b2": _ZEROS((1,), dtype=_FLOAT64),
    }
    for value in parameters.values():
        value.setflags(write=False)
    return parameters


def _stage_seeds(stage: int) -> tuple[int, ...]:
    if stage == 0:
        return INITIALIZATION_SEEDS[:1]
    if stage == 1:
        return INITIALIZATION_SEEDS
    raise EvalError(f"unknown stage {stage}")


def _copy_state(raw_state: object, context: str) -> OptimizerState:
    if type(raw_state) is not dict:
        raise EvalError(f"{context}: state must be an exact dict")
    raw_items = tuple(raw_state.items())
    snapshot: OptimizerState = {}
    for raw_key, raw_value in raw_items:
        if type(raw_key) is not str or not raw_key:
            raise EvalError(f"{context}: state keys must be non-empty exact strings")
        if type(raw_value) is float:
            snapshot[raw_key] = raw_value
            continue
        if type(raw_value) is not _NDARRAY_TYPE:
            raise EvalError(
                f"{context}: state value {raw_key!r} must be a NumPy array or float"
            )
        value_snapshot = _ARRAY(raw_value, copy=True, subok=False)
        value_snapshot.setflags(write=False)
        snapshot[raw_key] = value_snapshot
    return snapshot


def _validate_state(state: OptimizerState, context: str) -> None:
    for key, value in state.items():
        if type(value) is float:
            if not _ISFINITE_FLOAT(value):
                raise EvalError(f"{context}: state value {key!r} is non-finite")
            continue
        try:
            finite = bool(_ALL(_ISFINITE(value)))
        except TypeError as exc:
            raise EvalError(
                f"{context}: state array {key!r} must have a finite numeric dtype"
            ) from exc
        if not finite:
            raise EvalError(f"{context}: state array {key!r} is non-finite")


def _state_snapshot(raw_state: object, context: str) -> OptimizerState:
    snapshot = _copy_state(raw_state, context)
    _validate_state(snapshot, context)
    return snapshot


def _state_for_candidate(state: OptimizerState) -> OptimizerState:
    candidate_state: OptimizerState = {}
    for key, value in state.items():
        if type(value) is float:
            candidate_state[key] = value
        else:
            value_copy = _ARRAY(value, copy=True, subok=False)
            value_copy.setflags(write=False)
            candidate_state[key] = value_copy
    return candidate_state


def _normalize_update(
    raw_result: object,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    context: str,
) -> tuple[np.ndarray, OptimizerState]:
    if type(raw_result) is not tuple:
        raise EvalError(f"{context}: update must return an exact tuple")
    try:
        raw_parameter, raw_state = raw_result
    except ValueError as exc:
        raise EvalError(f"{context}: update must return exactly two values") from exc
    if type(raw_parameter) is not _NDARRAY_TYPE:
        raise EvalError(f"{context}: new parameter must be an exact NumPy array")

    parameter_snapshot = _ARRAY(raw_parameter, copy=True, subok=False)
    parameter_snapshot.setflags(write=False)
    state_snapshot = _state_snapshot(raw_state, context)
    if parameter_snapshot.shape != expected_shape:
        raise EvalError(
            f"{context}: parameter shape changed from {expected_shape} "
            f"to {parameter_snapshot.shape}"
        )
    if parameter_snapshot.dtype.str != expected_dtype:
        raise EvalError(
            f"{context}: parameter dtype changed from {expected_dtype} "
            f"to {parameter_snapshot.dtype.str}"
        )
    try:
        finite = bool(_ALL(_ISFINITE(parameter_snapshot)))
    except TypeError as exc:
        raise EvalError(f"{context}: parameter must have a finite numeric dtype") from exc
    if not finite:
        raise EvalError(f"{context}: new parameter contains a non-finite value")
    return parameter_snapshot, state_snapshot


def _forward(parameters: Parameters, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = _TANH(features @ parameters["w1"] + parameters["b1"])
    logits = hidden @ parameters["w2"] + parameters["b2"]
    return hidden, logits


def _loss(logits: np.ndarray, labels: np.ndarray) -> float:
    with _ERRSTATE(over="ignore", invalid="ignore"):
        losses = _LOGADDEXP(0.0, logits) - labels * logits
    return float(_MEAN(losses))


def _loss_and_gradients(
    parameters: Parameters,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, Parameters]:
    with _ERRSTATE(over="ignore", invalid="ignore"):
        hidden, logits = _forward(parameters, features)
        loss = _loss(logits, labels)
        probabilities = 1.0 / (1.0 + _EXP(-_CLIP(logits, -60.0, 60.0)))
        output_delta = (probabilities - labels) / features.shape[0]
        grad_w2 = hidden.T @ output_delta
        grad_b2 = _SUM(output_delta, axis=0)
        hidden_delta = (output_delta @ parameters["w2"].T) * (1.0 - hidden * hidden)
        grad_w1 = features.T @ hidden_delta
        grad_b1 = _SUM(hidden_delta, axis=0)
    gradients = {
        "w1": _ARRAY(grad_w1, dtype=_FLOAT64, copy=True),
        "b1": _ARRAY(grad_b1, dtype=_FLOAT64, copy=True),
        "w2": _ARRAY(grad_w2, dtype=_FLOAT64, copy=True),
        "b2": _ARRAY(grad_b2, dtype=_FLOAT64, copy=True),
    }
    for value in gradients.values():
        value.setflags(write=False)
    return loss, gradients


def _initialize_states(init_state: InitState, parameters: Parameters) -> dict[str, OptimizerState]:
    states: dict[str, OptimizerState] = {}
    for name in PARAMETER_NAMES:
        shape = tuple(parameters[name].shape)
        context = f"step 0 parameter {name}"
        try:
            raw_state = init_state(shape)
        except Exception as exc:
            raise EvalError(
                f"{context}: init_state raised {type(exc).__name__}: {exc}"
            ) from exc
        states[name] = _state_snapshot(raw_state, context)
    return states


def _apply_updates(
    update: UpdateRule,
    parameters: Parameters,
    gradients: Parameters,
    states: dict[str, OptimizerState],
    step: int,
) -> tuple[Parameters, dict[str, OptimizerState]]:
    next_parameters: Parameters = {}
    next_states: dict[str, OptimizerState] = {}
    for name in PARAMETER_NAMES:
        parameter = parameters[name]
        gradient = gradients[name]
        expected_shape = tuple(parameter.shape)
        expected_dtype = parameter.dtype.str
        parameter_input = _ARRAY(parameter, copy=True, subok=False)
        gradient_input = _ARRAY(gradient, copy=True, subok=False)
        parameter_input.setflags(write=False)
        gradient_input.setflags(write=False)
        context = f"step {step} parameter {name}"
        try:
            raw_result = update(
                parameter_input,
                gradient_input,
                _state_for_candidate(states[name]),
                step,
            )
        except Exception as exc:
            raise EvalError(
                f"{context}: update raised {type(exc).__name__}: {exc}"
            ) from exc
        next_parameter, next_state = _normalize_update(
            raw_result,
            expected_shape,
            expected_dtype,
            context,
        )
        next_parameters[name] = next_parameter
        next_states[name] = next_state
    return next_parameters, next_states


def _train_once(
    init_state: InitState,
    update: UpdateRule,
    prepared_run: PreparedRun,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    stage: int,
) -> tuple[float, float, float]:
    seed, initial_parameters = prepared_run
    parameters = {
        name: _ARRAY(initial_parameters[name], copy=True, subok=False)
        for name in PARAMETER_NAMES
    }
    for value in parameters.values():
        value.setflags(write=False)
    initial_logits = _forward(parameters, train_x)[1]
    initial_loss = _loss(initial_logits, train_y)
    if not _ISFINITE_FLOAT(initial_loss):
        raise EvalError(
            f"stage {stage} seed {seed} initial training loss is non-finite: "
            f"initial={initial_loss!r}"
        )

    states = _initialize_states(init_state, parameters)
    for step in range(1, STEPS + 1):
        _, gradients = _loss_and_gradients(parameters, train_x, train_y)
        parameters, states = _apply_updates(update, parameters, gradients, states, step)

    final_train_logits = _forward(parameters, train_x)[1]
    validation_logits = _forward(parameters, validation_x)[1]
    final_train_loss = _loss(final_train_logits, train_y)
    validation_loss = _loss(validation_logits, validation_y)
    if not _ISFINITE_FLOAT(final_train_loss):
        raise EvalError(
            f"stage {stage} seed {seed} final training loss is non-finite: "
            f"initial={initial_loss!r}, final={final_train_loss!r}"
        )
    if final_train_loss >= initial_loss:
        raise EvalError(
            f"stage {stage} seed {seed} training loss did not strictly decrease: "
            f"initial={initial_loss!r}, final={final_train_loss!r}"
        )
    if not _ISFINITE_FLOAT(validation_loss):
        raise EvalError(
            f"stage {stage} seed {seed} validation loss is non-finite: "
            f"val_loss={validation_loss!r}"
        )
    predictions = validation_logits >= 0.0
    validation_accuracy = float(_MEAN(predictions == validation_y))
    return final_train_loss, validation_loss, validation_accuracy


def _mean(values: list[float]) -> float:
    return _FSUM(values) / len(values)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Train the fixed MLP and score the candidate optimizer rule."""

    seeds = _stage_seeds(stage)
    train_x, train_y, validation_x, validation_y = _load_data()
    prepared_runs = [(seed, _initial_parameters(seed)) for seed in seeds]
    init_state, update = _load_candidate(candidate_dir)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    for prepared_run in prepared_runs:
        train_loss, validation_loss, validation_accuracy = _train_once(
            init_state,
            update,
            prepared_run,
            train_x,
            train_y,
            validation_x,
            validation_y,
            stage,
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    return {
        GATE: 1.0,
        "val_loss": _mean(validation_losses),
        "train_loss": _mean(train_losses),
        "val_accuracy": _mean(validation_accuracies),
        "steps": float(STEPS),
        **source_metrics(candidate_dir, "rule.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no ceiling because the synthetic task's Bayes error is unknown."""

    return None


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
