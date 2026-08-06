"""Exact evaluator for Low Autocorrelation Binary Sequences."""

from __future__ import annotations

import importlib.util
import inspect
import operator
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec

GATE = "valid_binary_sequence"
METRIC = "merit_factor"
MAXIMIZE = True

DESCRIPTORS = [
    {
        "name": "positive_fraction",
        "metric": "positive_fraction",
        "bins": 12,
        "lo": 0.0,
        "hi": 1.0,
    },
    {
        "name": "normalized_max_abs_autocorrelation",
        "metric": "normalized_max_abs_autocorrelation",
        "bins": 12,
        "lo": 0.0,
        "hi": 1.0,
    },
]

_CELLS = {
    "n13-validation": (13, 130_013, 30.0),
    "n41-calibration": (41, 410_041, 300.0),
    "n61-calibration": (61, 610_061, 300.0),
    "n71-frontier": (71, 710_071, 300.0),
    "n81-frontier": (81, 810_081, 300.0),
    "n91-frontier": (91, 910_091, 300.0),
    "n101-frontier": (101, 1_010_101, 300.0),
    "n121-frontier": (121, 1_210_121, 300.0),
}
_CELL = os.environ.get("AUTOEVOLVE_CELL", "n13-validation")
if _CELL not in _CELLS:
    _choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL!r}")

SEQUENCE_LENGTH, CELL_SEED, _STAGE_TIMEOUT_S = _CELLS[_CELL]
STAGES: list[StageSpec] = [
    StageSpec(name="search-and-exact-autocorrelation-gate", timeout_s=_STAGE_TIMEOUT_S),
]
_DEADLINE_HEADROOM_S = 3.0

# Candidate import happens before normalization, verification, and scoring. Bind
# trusted primitives now so candidate code cannot replace a later gate dependency.
_ABS = abs
_ANY = any
_BOOL = bool
_CALLABLE = callable
_DICT_POP = dict.pop
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_HASH = hash
_INDEX = operator.index
_INT = int
_ITER = iter
_LEN = len
_LIST = list
_MAX = max
_MODULE_FROM_SPEC = importlib.util.module_from_spec
_MODULES = sys.modules
_MONOTONIC = time.monotonic
_NEXT = next
_PARAMETER = inspect.Parameter
_POSITIONAL_ONLY = _PARAMETER.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD = _PARAMETER.POSITIONAL_OR_KEYWORD
_KEYWORD_ONLY = _PARAMETER.KEYWORD_ONLY
_RANGE = range
_SET = set
_SIGNATURE = inspect.signature
_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_STOP_ITERATION = StopIteration
_STR = str
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_VAR_KEYWORD = _PARAMETER.VAR_KEYWORD
_VAR_POSITIONAL = _PARAMETER.VAR_POSITIONAL


def _snapshot_sequence(raw: object) -> tuple[int, ...]:
    """Consume the candidate result once and retain exact plain integers."""

    try:
        iterator = _ITER(raw)
    except _TYPE_ERROR as exc:
        raise EvalError(
            f"solve() result must be iterable, got {_TYPE(raw).__name__}"
        ) from exc

    items: list[object] = []
    while _LEN(items) <= SEQUENCE_LENGTH:
        try:
            item = _NEXT(iterator)
        except _STOP_ITERATION:
            break
        except _EXCEPTION as exc:
            raise EvalError(
                f"solve() result failed while reading entry {_LEN(items)}: {exc}"
            ) from exc
        if _LEN(items) == SEQUENCE_LENGTH:
            raise EvalError(
                f"solve() result must contain exactly {SEQUENCE_LENGTH} entries; got more"
            )
        items.append(item)

    if _LEN(items) != SEQUENCE_LENGTH:
        raise EvalError(
            f"solve() result must contain exactly {SEQUENCE_LENGTH} entries; "
            f"got {_LEN(items)}"
        )

    normalized: list[int] = []
    for index, raw_value in _ENUMERATE(items):
        if _TYPE(raw_value) is _BOOL:
            raise EvalError(f"sequence entry {index} must be -1 or +1, got bool")
        try:
            value = _INT(_INDEX(raw_value))
        except _EXCEPTION as exc:
            raise EvalError(
                f"sequence entry {index} must be an integer -1 or +1, "
                f"got {_TYPE(raw_value).__name__}"
            ) from exc
        if value != -1 and value != 1:
            raise EvalError(f"sequence entry {index} must be -1 or +1, got {value}")
        normalized.append(value)
    return _TUPLE(normalized)


def _call_solver(solver: Callable[..., object], deadline: float) -> object:
    try:
        parameters = _LIST(_SIGNATURE(solver).parameters.values())
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"could not inspect solve() signature: {exc}") from exc

    positional_kinds = (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL)
    if not _ANY(parameter.kind in positional_kinds for parameter in parameters):
        raise EvalError("solve() must accept n as a positional argument")

    args: list[object] = [SEQUENCE_LENGTH]
    kwargs: dict[str, object] = {}
    optional_values = {"deadline": deadline, "seed": CELL_SEED}
    explicitly_bound: set[str] = _SET()

    for parameter in parameters:
        if parameter.name not in optional_values:
            continue
        value = optional_values[parameter.name]
        if parameter.kind is _POSITIONAL_ONLY:
            args.append(value)
            explicitly_bound.add(parameter.name)
        elif parameter.kind in (_POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY):
            kwargs[parameter.name] = value
            explicitly_bound.add(parameter.name)

    accepts_var_positional = _ANY(
        parameter.kind is _VAR_POSITIONAL for parameter in parameters
    )
    accepts_var_keyword = _ANY(parameter.kind is _VAR_KEYWORD for parameter in parameters)
    for name in ("deadline", "seed"):
        if name in explicitly_bound:
            continue
        if accepts_var_keyword:
            kwargs[name] = optional_values[name]
        elif accepts_var_positional:
            args.append(optional_values[name])

    try:
        return solver(*args, **kwargs)
    except _EXCEPTION as exc:
        raise EvalError(f"solve() raised: {exc}") from exc


def _load_candidate_sequence(candidate_dir: Path, deadline: float) -> tuple[int, ...]:
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise EvalError("candidate is missing solver.py")

    module_name = f"_autoevolve_labs_{_ABS(_HASH(_STR(path.resolve())))}"
    spec = _SPEC_FROM_FILE_LOCATION(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load candidate solver.py")
    module: ModuleType = _MODULE_FROM_SPEC(spec)
    _MODULES[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except _EXCEPTION as exc:
            raise EvalError(f"solver.py failed to import: {exc}") from exc
        solver = _GETATTR(module, "solve", None)
        if not _CALLABLE(solver):
            raise EvalError("solver.py must define callable solve()")
        raw = _call_solver(solver, deadline)
        return _snapshot_sequence(raw)
    finally:
        _DICT_POP(_MODULES, module_name, None)


def _exact_autocorrelations(sequence: tuple[int, ...]) -> tuple[int, ...]:
    correlations: list[int] = []
    for lag in _RANGE(1, SEQUENCE_LENGTH):
        correlation = 0
        for index in _RANGE(SEQUENCE_LENGTH - lag):
            correlation += sequence[index] * sequence[index + lag]
        correlations.append(correlation)
    return _TUPLE(correlations)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run candidate search, enforce the exact gate, and derive every metric."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    candidate_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if candidate_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")

    deadline = _MONOTONIC() + candidate_budget
    sequence = _load_candidate_sequence(candidate_dir, deadline)
    correlations = _exact_autocorrelations(sequence)

    energy = 0
    max_abs_autocorrelation = 0
    for correlation in correlations:
        energy += correlation * correlation
        max_abs_autocorrelation = _MAX(max_abs_autocorrelation, _ABS(correlation))
    if energy <= 0:
        raise EvalError("exact energy must be positive for the selected sequence length")

    positive_count = 0
    for value in sequence:
        if value == 1:
            positive_count += 1

    return {
        GATE: 1.0,
        METRIC: _FLOAT(SEQUENCE_LENGTH * SEQUENCE_LENGTH) / _FLOAT(2 * energy),
        "energy": _FLOAT(energy),
        "length": _FLOAT(SEQUENCE_LENGTH),
        "max_abs_autocorrelation": _FLOAT(max_abs_autocorrelation),
        "positive_fraction": _FLOAT(positive_count) / _FLOAT(SEQUENCE_LENGTH),
        "normalized_max_abs_autocorrelation": _FLOAT(max_abs_autocorrelation)
        / _FLOAT(SEQUENCE_LENGTH),
    }


def ceiling() -> dict[str, float | str] | None:
    """No stored merit-factor ceiling participates in this campaign."""

    return None
