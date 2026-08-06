"""Exact geometric evaluator for equal-circle packing point sets."""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

GATE = "coordinates_valid"
METRIC = "min_pairwise_distance"
MAXIMIZE = True

DESCRIPTORS = [
    {
        "name": "boundary_point_count",
        "metric": "boundary_point_count",
        "bins": 16,
        "lo": 0.0,
        "hi": 62.0,
    },
    {
        "name": "contact_pair_fraction",
        "metric": "contact_pair_fraction",
        "bins": 12,
        "lo": 0.0,
        "hi": 1.0,
    },
]

# This tolerance only absorbs representation-level drift. Accepted values in the
# tolerance band are clamped before scoring so the allowance cannot improve spread.
CONTAINMENT_TOLERANCE = 1.0e-12
DESCRIPTOR_TOLERANCE = 1.0e-7
_DEADLINE_HEADROOM_S = 3.0

_CELLS = {
    "n2-validation": (2, 15.0, 2_001_003),
    "n10-calibration": (10, 300.0, 10_001_019),
    "n20-calibration": (20, 300.0, 20_001_021),
    "n30-calibration": (30, 300.0, 30_001_027),
    "n31-frontier": (31, 300.0, 31_001_033),
    "n37-frontier": (37, 300.0, 37_001_039),
    "n43-frontier": (43, 300.0, 43_001_047),
    "n51-frontier": (51, 300.0, 51_001_061),
    "n62-frontier": (62, 300.0, 62_001_067),
}
_CELL = os.environ.get("AUTOEVOLVE_CELL", "n2-validation")
if _CELL not in _CELLS:
    _choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL!r}")
POINT_COUNT, _STAGE_TIMEOUT_S, SEED = _CELLS[_CELL]

STAGES: list[StageSpec] = [
    StageSpec(name="search-and-exact-geometry-gate", timeout_s=_STAGE_TIMEOUT_S),
]

# Candidate import happens before normalization and verification. Bind trusted
# primitives now so candidate code cannot replace a later gate dependency.
_ABS = abs
_BOOL = bool
_CALLABLE = callable
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_HASH = hash
_INT = int
_ISFINITE = math.isfinite
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_LIST = list
_MAPPING_TYPE = Mapping
_MAX = max
_MIN = min
_MONOTONIC = time.monotonic
_NEXT = next
_OVERFLOW_ERROR = OverflowError
_PARAMETER_EMPTY = inspect.Parameter.empty
_POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
_SIGNATURE = inspect.signature
_SQRT = math.sqrt
_STOP_ITERATION = StopIteration
_TEXT_TYPES = (str, bytes, bytearray)
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_NP_ANY = np.any
_NP_ARRAY = np.array
_NP_COUNT_NONZERO = np.count_nonzero
_NP_EINSUM = np.einsum
_NP_FLOAT64 = np.float64
_NP_MIN = np.min
_NP_TRIU_INDICES = np.triu_indices

_PAIR_INDICES = _NP_TRIU_INDICES(POINT_COUNT, k=1)
_PAIR_COUNT = POINT_COUNT * (POINT_COUNT - 1) // 2


def _snapshot_sequence(raw: object, field: str, limit: int) -> tuple[object, ...]:
    """Consume candidate-controlled iteration once with a strict item cap."""

    if _ISINSTANCE(raw, _TEXT_TYPES) or _ISINSTANCE(raw, _MAPPING_TYPE):
        raise EvalError(f"{field} must be an array-like iterable")
    try:
        iterator = _ITER(raw)
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be iterable, got {_TYPE(raw).__name__}") from exc

    items: list[object] = []
    while _LEN(items) <= limit:
        try:
            item = _NEXT(iterator)
        except _STOP_ITERATION:
            return _TUPLE(items)
        except _EXCEPTION as exc:
            raise EvalError(f"{field} failed while reading item {_LEN(items)}: {exc}") from exc
        if _LEN(items) == limit:
            raise EvalError(f"{field} may contain at most {limit} items")
        items.append(item)
    raise EvalError(f"{field} exceeded its normalization limit")


def _finite_coordinate(raw: object, field: str) -> float:
    if _TYPE(raw) is _BOOL:
        raise EvalError(f"{field} must be a finite real coordinate, got bool")
    try:
        value = _FLOAT(raw)
    except (_TYPE_ERROR, _VALUE_ERROR, _OVERFLOW_ERROR) as exc:
        raise EvalError(f"{field} must be a finite real coordinate") from exc
    if not _ISFINITE(value):
        raise EvalError(f"{field} must be finite")
    return value


def _normalize_points(raw: object) -> tuple[tuple[float, float], ...]:
    raw_points = _snapshot_sequence(raw, "solve() result", POINT_COUNT + 1)
    if _LEN(raw_points) != POINT_COUNT:
        raise EvalError(f"solve() result must contain exactly {POINT_COUNT} points")

    points: list[tuple[float, float]] = []
    for point_index, raw_point in _ENUMERATE(raw_points):
        coordinates = _snapshot_sequence(raw_point, f"point {point_index}", 3)
        if _LEN(coordinates) != 2:
            raise EvalError(f"point {point_index} must contain exactly two coordinates")
        x = _finite_coordinate(coordinates[0], f"point {point_index} x")
        y = _finite_coordinate(coordinates[1], f"point {point_index} y")
        points.append((x, y))
    return _TUPLE(points)


def _check_and_clamp_containment(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Check each axis independently and remove tolerance-only outward drift."""

    checked: list[tuple[float, float]] = []
    for point_index, point in _ENUMERATE(points):
        normalized: list[float] = []
        for axis_index, coordinate in _ENUMERATE(point):
            if (
                coordinate < -CONTAINMENT_TOLERANCE
                or coordinate > 1.0 + CONTAINMENT_TOLERANCE
            ):
                axis = "x" if axis_index == 0 else "y"
                raise EvalError(
                    f"point {point_index} {axis} coordinate {coordinate!r} is outside "
                    "the closed unit square"
                )
            normalized.append(_MIN(1.0, _MAX(0.0, coordinate)))
        checked.append((normalized[0], normalized[1]))
    return _TUPLE(checked)


def _geometry_metrics(points: tuple[tuple[float, float], ...]) -> dict[str, float]:
    array = _NP_ARRAY(points, dtype=_NP_FLOAT64)
    deltas = array[:, None, :] - array[None, :, :]
    squared_matrix = _NP_EINSUM("ijk,ijk->ij", deltas, deltas)
    pair_squared = squared_matrix[_PAIR_INDICES]
    minimum_squared = _MAX(0.0, _FLOAT(_NP_MIN(pair_squared)))
    minimum_distance = _SQRT(minimum_squared)

    boundary_mask = _NP_ANY(
        (array <= DESCRIPTOR_TOLERANCE) | (array >= 1.0 - DESCRIPTOR_TOLERANCE),
        axis=1,
    )
    boundary_point_count = _INT(_NP_COUNT_NONZERO(boundary_mask))
    contact_limit_squared = (minimum_distance + DESCRIPTOR_TOLERANCE) ** 2
    contact_count = _INT(_NP_COUNT_NONZERO(pair_squared <= contact_limit_squared))
    contact_pair_fraction = contact_count / _PAIR_COUNT
    circle_radius = minimum_distance / (2.0 * (1.0 + minimum_distance))

    return {
        GATE: 1.0,
        METRIC: minimum_distance,
        "circle_radius": circle_radius,
        "point_count": _FLOAT(POINT_COUNT),
        "pair_count": _FLOAT(_PAIR_COUNT),
        "boundary_point_count": _FLOAT(boundary_point_count),
        "contact_pair_fraction": contact_pair_fraction,
    }


def _call_solver(solver: object, deadline: float) -> object:
    try:
        parameters = _LIST(_SIGNATURE(solver).parameters.values())
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"could not inspect solve() signature: {exc}") from exc

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD)
    ]
    var_positional = _NEXT(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _VAR_POSITIONAL
        ),
        None,
    )
    var_keyword = _NEXT(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _VAR_KEYWORD
        ),
        None,
    )
    if not positional and var_positional is None:
        raise EvalError("solve() must accept n as a positional argument")

    consumed_n = positional[0] if positional else None
    args: list[object] = [POINT_COUNT]
    kwargs: dict[str, object] = {}
    supplied = {"deadline": False, "seed": False}
    values: dict[str, object] = {"deadline": deadline, "seed": SEED}

    for parameter in parameters:
        if parameter is consumed_n or parameter.name not in values:
            continue
        value = values[parameter.name]
        if parameter.kind is _POSITIONAL_ONLY:
            args.append(value)
        elif parameter.kind in (
            _POSITIONAL_OR_KEYWORD,
            _KEYWORD_ONLY,
        ):
            kwargs[parameter.name] = value
        supplied[parameter.name] = True

    missing = [name for name, was_supplied in supplied.items() if not was_supplied]
    if var_positional is not None:
        for name in missing:
            args.append(values[name])
            supplied[name] = True
    elif var_keyword is not None:
        for name in missing:
            kwargs[name] = values[name]
            supplied[name] = True

    for parameter in parameters:
        if parameter is consumed_n or parameter.name in values:
            continue
        if parameter.kind in (
            _POSITIONAL_ONLY,
            _POSITIONAL_OR_KEYWORD,
            _KEYWORD_ONLY,
        ) and parameter.default is _PARAMETER_EMPTY:
            raise EvalError(f"solve() has unsupported required parameter {parameter.name!r}")

    try:
        return solver(*args, **kwargs)
    except _EXCEPTION as exc:
        raise EvalError(f"solve() raised: {exc}") from exc


def _load_candidate_points(candidate_dir: Path, deadline: float) -> tuple[tuple[float, float], ...]:
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise EvalError("candidate is missing solver.py")
    module_name = f"_autoevolve_circlepack_{_ABS(_HASH(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load candidate solver.py")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except _EXCEPTION as exc:
            raise EvalError(f"solver.py failed to import: {exc}") from exc
        solver = _GETATTR(module, "solve", None)
        if not _CALLABLE(solver):
            raise EvalError("solver.py must define callable solve()")
        return _normalize_points(_call_solver(solver, deadline))
    finally:
        sys.modules.pop(module_name, None)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run one candidate search, validate its points, and recompute all metrics."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    started = _MONOTONIC()
    candidate_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if candidate_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")

    points = _load_candidate_points(candidate_dir, started + candidate_budget)
    contained_points = _check_and_clamp_containment(points)
    return _geometry_metrics(contained_points)


def ceiling() -> dict[str, float | str] | None:
    """No common evaluator-computed ceiling is declared for these cells."""

    return None
