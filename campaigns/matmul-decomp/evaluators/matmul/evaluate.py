"""Gate low-rank decompositions of small matrix multiplication tensors."""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

GATE = "tensor_identity"
METRIC = "rank"
MAXIMIZE = False

DESCRIPTORS = [
    {
        "name": "coefficient_sparsity",
        "metric": "coefficient_sparsity",
        "bins": 10,
        "lo": 0.0,
        "hi": 1.0,
    },
    {
        "name": "distinct_coefficient_values",
        "metric": "distinct_coefficient_values",
        "bins": 16,
        "lo": 1.0,
        "hi": 64.0,
    },
]

_DEADLINE_HEADROOM_S = 5.0
_EXACT_NUMERATORS = (-2, -1, 0, 1, 2)
_EXACT_DENOMINATOR = 2


@dataclass(frozen=True)
class CellSpec:
    key: str
    m: int
    k: int
    n: int
    field: str
    coefficient_mode: str
    target_rank: int
    timeout_s: float
    seed: int
    allowed_numerators: tuple[int, ...] = ()
    denominator: int = 1

    @property
    def naive_rank(self) -> int:
        return self.m * self.k * self.n


_CELLS = {
    "2x2-real-r7-validation": CellSpec(
        key="2x2-real-r7-validation",
        m=2,
        k=2,
        n=2,
        field="real",
        coefficient_mode="exact",
        target_rank=7,
        timeout_s=120.0,
        seed=2207,
        allowed_numerators=_EXACT_NUMERATORS,
        denominator=_EXACT_DENOMINATOR,
    ),
    "3x3-real-r23-frontier": CellSpec(
        key="3x3-real-r23-frontier",
        m=3,
        k=3,
        n=3,
        field="real",
        coefficient_mode="numeric",
        target_rank=23,
        timeout_s=600.0,
        seed=3323,
    ),
    "4x4-complex-r48-frontier": CellSpec(
        key="4x4-complex-r48-frontier",
        m=4,
        k=4,
        n=4,
        field="complex",
        coefficient_mode="exact",
        target_rank=48,
        timeout_s=600.0,
        seed=4448,
        allowed_numerators=_EXACT_NUMERATORS,
        denominator=_EXACT_DENOMINATOR,
    ),
}

_CELL_KEY = os.environ.get("AUTOEVOLVE_CELL", "2x2-real-r7-validation")
if _CELL_KEY not in _CELLS:
    _choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL_KEY!r}")
CELL = _CELLS[_CELL_KEY]

STAGES: list[StageSpec] = [
    StageSpec(name=f"{CELL.key}-search-and-gate", timeout_s=CELL.timeout_s),
]

# Candidate import happens before normalization and verification. Bind every
# trusted primitive used after import so candidate code cannot replace a later
# gate dependency through builtins or the shared NumPy module.
_ABS = abs
_ANY = any
_ARGWHERE = np.argwhere
_ARRAY = np.array
_ARRAY_EQUAL = np.array_equal
_BOOL = bool
_CALLABLE = callable
_COMPLEX = complex
_ENUMERATE = enumerate
_EINSUM = np.einsum
_EXCEPTION = Exception
_FLOAT = float
_FLOAT_EPSILON = np.finfo(np.float64).eps
_FRACTION = Fraction
_FRACTION_FROM_FLOAT = Fraction.from_float
_GETATTR = getattr
_HASH = hash
_INT = int
_ISFINITE = math.isfinite
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_LIST = list
_MAPPING = Mapping
_MAX = max
_MAXIMUM = np.maximum
_MONOTONIC = time.monotonic
_NEXT = next
_NP_BOOL = np.bool_
_NP_COMPLEX128 = np.complex128
_NP_COMPLEX = np.complexfloating
_NP_FLOAT64 = np.float64
_NP_FLOAT = np.floating
_NP_INTEGER = np.integer
_NP_INT64 = np.int64
_PARAM_POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
_PARAM_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_PARAM_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
_PARAM_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_RANGE = range
_SET = set
_SIGNATURE = inspect.signature
_SORTED = sorted
_STOP_ITERATION = StopIteration
_STR = str
_SUM = sum
_TEXT_TYPES = (str, bytes, bytearray)
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_ZEROS = np.zeros
_MODULES = sys.modules
_DICT_POP = dict.pop


def _snapshot_sequence(raw: object, field: str, limit: int) -> tuple[object, ...]:
    if _ISINSTANCE(raw, _TEXT_TYPES) or _ISINSTANCE(raw, _MAPPING):
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


def _mapping_snapshot(raw: object) -> dict[str, object]:
    if not _ISINSTANCE(raw, _MAPPING):
        raise EvalError(f"solve() must return a mapping, got {_TYPE(raw).__name__}")
    try:
        raw_items = raw.items()
    except _EXCEPTION as exc:
        raise EvalError(f"solve() result could not expose items once: {exc}") from exc
    items = _snapshot_sequence(raw_items, "solve() result items", 3)
    snapshot: dict[str, object] = {}
    for index, item in _ENUMERATE(items):
        pair = _snapshot_sequence(item, f"solve() result item {index}", 2)
        if _LEN(pair) != 2:
            raise EvalError(f"solve() result item {index} must be a key-value pair")
        key, value = pair
        if _TYPE(key) is not _STR:
            raise EvalError(f"solve() result key {index} must be a plain string")
        if key in snapshot:
            raise EvalError(f"solve() result contains duplicate key {key!r}")
        snapshot[key] = value
    expected = {"U", "V", "W"}
    found = _SET(snapshot)
    if found != expected:
        missing = _SORTED(expected - found)
        extra = _SORTED(found - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise EvalError(f"decomposition schema is exact; {'; '.join(details)}")
    return snapshot


def _snapshot_matrix(
    raw: object,
    field: str,
    width: int,
    max_rank: int,
) -> tuple[tuple[object, ...], ...]:
    rows = _snapshot_sequence(raw, field, max_rank)
    normalized: list[tuple[object, ...]] = []
    for row_index, raw_row in _ENUMERATE(rows):
        row = _snapshot_sequence(raw_row, f"{field}[{row_index}]", width)
        if _LEN(row) != width:
            raise EvalError(
                f"{field}[{row_index}] must contain exactly {width} coefficients; "
                f"got {_LEN(row)}"
            )
        normalized.append(row)
    return _TUPLE(normalized)


def _real_fraction(raw: object, field: str) -> Fraction:
    if _TYPE(raw) is _BOOL or _ISINSTANCE(raw, _NP_BOOL):
        raise EvalError(f"{field} must be a coefficient, got bool")
    if _ISINSTANCE(raw, _FRACTION):
        return _FRACTION(raw.numerator, raw.denominator)
    if _ISINSTANCE(raw, (_INT, _NP_INTEGER)):
        return _FRACTION(_INT(raw), 1)
    if _ISINSTANCE(raw, (_FLOAT, _NP_FLOAT)):
        value = _FLOAT(raw)
        if not _ISFINITE(value):
            raise EvalError(f"{field} must be finite")
        return _FRACTION_FROM_FLOAT(value)
    raise EvalError(f"{field} has unsupported coefficient type {_TYPE(raw).__name__}")


def _exact_coefficient(raw: object, field: str, cell: CellSpec) -> tuple[int, int]:
    if _ISINSTANCE(raw, (_COMPLEX, _NP_COMPLEX)):
        value = _COMPLEX(raw)
        if not _ISFINITE(value.real) or not _ISFINITE(value.imag):
            raise EvalError(f"{field} must be finite")
        real = _FRACTION_FROM_FLOAT(value.real)
        imag = _FRACTION_FROM_FLOAT(value.imag)
    else:
        real = _real_fraction(raw, field)
        imag = _FRACTION(0, 1)

    if cell.field == "real" and imag != 0:
        raise EvalError(f"{field} must be real for cell {cell.key}")

    scaled_parts: list[int] = []
    for component_name, component in (("real", real), ("imaginary", imag)):
        scaled = component * cell.denominator
        if scaled.denominator != 1 or scaled.numerator not in cell.allowed_numerators:
            allowed = ", ".join(
                _STR(_FRACTION(value, cell.denominator))
                for value in cell.allowed_numerators
            )
            raise EvalError(
                f"{field} {component_name} component is outside the declared "
                f"discrete set {{{allowed}}}; exact coefficients are never rounded"
            )
        scaled_parts.append(_INT(scaled.numerator))
    return scaled_parts[0], scaled_parts[1]


def _numeric_coefficient(raw: object, field: str, cell: CellSpec) -> float | complex:
    if _TYPE(raw) is _BOOL or _ISINSTANCE(raw, _NP_BOOL):
        raise EvalError(f"{field} must be a coefficient, got bool")
    if _ISINSTANCE(raw, _FRACTION):
        value: float | complex = _FLOAT(raw)
    elif _ISINSTANCE(raw, (_INT, _FLOAT, _NP_INTEGER, _NP_FLOAT)):
        value = _FLOAT(raw)
    elif _ISINSTANCE(raw, (_COMPLEX, _NP_COMPLEX)):
        value = _COMPLEX(raw)
    else:
        raise EvalError(f"{field} has unsupported coefficient type {_TYPE(raw).__name__}")

    complex_value = _COMPLEX(value)
    if not _ISFINITE(complex_value.real) or not _ISFINITE(complex_value.imag):
        raise EvalError(f"{field} must be finite")
    if cell.field == "real":
        if complex_value.imag != 0.0:
            raise EvalError(f"{field} must be real for cell {cell.key}")
        return _FLOAT(complex_value.real)
    return complex_value


def _normalize_decomposition(
    raw: object,
    cell: CellSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _mapping_snapshot(raw)
    widths = {"U": cell.m * cell.k, "V": cell.k * cell.n, "W": cell.m * cell.n}
    snapshots = {
        name: _snapshot_matrix(values[name], name, width, cell.naive_rank)
        for name, width in widths.items()
    }
    ranks = {name: _LEN(matrix) for name, matrix in snapshots.items()}
    if ranks["U"] <= 0:
        raise EvalError("decomposition rank must be positive")
    if _LEN(_SET(ranks.values())) != 1:
        raise EvalError(
            "U, V, and W must have the same row count; "
            f"got U={ranks['U']}, V={ranks['V']}, W={ranks['W']}"
        )

    matrices: list[np.ndarray] = []
    for name in ("U", "V", "W"):
        rows = snapshots[name]
        if cell.coefficient_mode == "exact":
            parsed = [
                [
                    _exact_coefficient(value, f"{name}[{row_index}][{column_index}]", cell)
                    for column_index, value in _ENUMERATE(row)
                ]
                for row_index, row in _ENUMERATE(rows)
            ]
            matrices.append(_ARRAY(parsed, dtype=_NP_INT64))
        else:
            parsed = [
                [
                    _numeric_coefficient(
                        value,
                        f"{name}[{row_index}][{column_index}]",
                        cell,
                    )
                    for column_index, value in _ENUMERATE(row)
                ]
                for row_index, row in _ENUMERATE(rows)
            ]
            dtype = _NP_COMPLEX128 if cell.field == "complex" else _NP_FLOAT64
            matrices.append(_ARRAY(parsed, dtype=dtype))
    return matrices[0], matrices[1], matrices[2]


def _target_tensor(cell: CellSpec, dtype: object) -> np.ndarray:
    target = _ZEROS((cell.m * cell.k, cell.k * cell.n, cell.m * cell.n), dtype=dtype)
    for i in _RANGE(cell.m):
        for j in _RANGE(cell.k):
            for ell in _RANGE(cell.n):
                target[i * cell.k + j, j * cell.n + ell, i * cell.n + ell] = 1
    return target


def _einsum_rank_terms(left: np.ndarray, right: np.ndarray, output: np.ndarray) -> np.ndarray:
    return _EINSUM("ra,rb,rc->abc", left, right, output, optimize=False)


def _verify_exact(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell: CellSpec,
) -> float:
    u, v, w = matrices
    ur, ui = u[..., 0], u[..., 1]
    vr, vi = v[..., 0], v[..., 1]
    wr, wi = w[..., 0], w[..., 1]

    reconstructed_real = (
        _einsum_rank_terms(ur, vr, wr)
        - _einsum_rank_terms(ur, vi, wi)
        - _einsum_rank_terms(ui, vr, wi)
        - _einsum_rank_terms(ui, vi, wr)
    )
    reconstructed_imag = (
        _einsum_rank_terms(ur, vr, wi)
        + _einsum_rank_terms(ur, vi, wr)
        + _einsum_rank_terms(ui, vr, wr)
        - _einsum_rank_terms(ui, vi, wi)
    )
    target = _target_tensor(cell, _NP_INT64) * (cell.denominator**3)
    valid = _ARRAY_EQUAL(reconstructed_real, target) and not _ANY(reconstructed_imag.flat)
    if not valid:
        mismatch = (reconstructed_real != target) | (reconstructed_imag != 0)
        index = _ARGWHERE(mismatch)[0]
        a, b, c = (_INT(value) for value in index)
        raise EvalError(
            "tensor identity failed at "
            f"({a}, {b}, {c}): scaled expected {target[a, b, c]}, "
            f"got {reconstructed_real[a, b, c]} + "
            f"{reconstructed_imag[a, b, c]}i"
        )
    return 0.0


def _verify_numeric(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell: CellSpec,
) -> float:
    u, v, w = matrices
    reconstructed = _einsum_rank_terms(u, v, w)
    target_dtype = _NP_COMPLEX128 if cell.field == "complex" else _NP_FLOAT64
    target = _target_tensor(cell, target_dtype)
    absolute_terms = _einsum_rank_terms(_ABS(u), _ABS(v), _ABS(w))

    rank = u.shape[0]
    # A real term has two products plus accumulation. Sixteen steps per rank
    # conservatively covers the component operations in complex arithmetic.
    operations_per_rank = 16 if cell.field == "complex" else 3
    rounding_steps = _MAX(1, operations_per_rank * rank)
    denominator = 1.0 - rounding_steps * _FLOAT_EPSILON
    if denominator <= 0.0:
        raise EvalError("numeric forward-error bound is undefined for this rank")
    gamma = rounding_steps * _FLOAT_EPSILON / denominator
    local_scale = _MAXIMUM(absolute_terms, _ABS(reconstructed))
    local_scale = _MAXIMUM(local_scale, _ABS(target))
    tolerance = gamma * local_scale
    absolute_error = _ABS(reconstructed - target)
    failures = absolute_error > tolerance
    if _ANY(failures.flat):
        index = _ARGWHERE(failures)[0]
        a, b, c = (_INT(value) for value in index)
        raise EvalError(
            "numeric tensor identity failed at "
            f"({a}, {b}, {c}): error {_FLOAT(absolute_error[a, b, c]):.6g} "
            f"exceeds derived tolerance {_FLOAT(tolerance[a, b, c]):.6g}"
        )
    return _FLOAT(tolerance.max())


def _descriptor_metrics(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell: CellSpec,
) -> tuple[float, float, float]:
    total = _SUM(matrix.shape[0] * matrix.shape[1] for matrix in matrices)
    zero_count = 0
    distinct: set[object] = _SET()
    if cell.coefficient_mode == "exact":
        for matrix in matrices:
            for real, imag in matrix.reshape(-1, 2):
                pair = (_INT(real), _INT(imag))
                distinct.add(pair)
                if pair == (0, 0):
                    zero_count += 1
    else:
        for matrix in matrices:
            for value in matrix.flat:
                normalized = _COMPLEX(value) if cell.field == "complex" else _FLOAT(value)
                distinct.add(normalized)
                if normalized == 0:
                    zero_count += 1
    return zero_count / total, _FLOAT(_LEN(distinct)), _FLOAT(total)


def _candidate_payload(cell: CellSpec) -> dict[str, object]:
    return {
        "cell": cell.key,
        "m": cell.m,
        "k": cell.k,
        "n": cell.n,
        "field": cell.field,
        "coefficient_mode": cell.coefficient_mode,
        "target_rank": cell.target_rank,
        "seed": cell.seed,
        "allowed_numerators": cell.allowed_numerators,
        "coefficient_denominator": cell.denominator,
    }


def _call_solver(
    solver: Callable[..., object],
    payload: dict[str, object],
    deadline: float,
    seed: int,
) -> object:
    try:
        signature = _SIGNATURE(solver)
        parameters = _LIST(signature.parameters.values())
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"could not inspect solve() signature: {exc}") from exc

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (_PARAM_POSITIONAL_ONLY, _PARAM_POSITIONAL_OR_KEYWORD)
    ]
    var_positional = _NEXT(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _PARAM_VAR_POSITIONAL
        ),
        None,
    )
    var_keyword = _NEXT(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _PARAM_VAR_KEYWORD
        ),
        None,
    )
    if not positional and var_positional is None:
        raise EvalError("solve() must accept the problem mapping positionally")

    arguments: list[object] = [payload]
    keywords: dict[str, object] = {}
    supplied = _SET()
    first_positional = positional[0] if positional else None
    for parameter in parameters:
        if parameter is first_positional:
            continue
        if parameter.kind in (_PARAM_VAR_POSITIONAL, _PARAM_VAR_KEYWORD):
            continue
        if parameter.name not in {"deadline", "seed"}:
            if parameter.kind is _PARAM_POSITIONAL_ONLY:
                raise EvalError(
                    "solve() has an unsupported positional-only parameter "
                    f"{parameter.name!r} after the problem mapping"
                )
            continue
        value = deadline if parameter.name == "deadline" else seed
        if parameter.kind is _PARAM_POSITIONAL_ONLY:
            arguments.append(value)
        else:
            keywords[parameter.name] = value
        supplied.add(parameter.name)

    missing = [name for name in ("deadline", "seed") if name not in supplied]
    if missing and var_keyword is not None:
        for name in missing:
            keywords[name] = deadline if name == "deadline" else seed
        missing = []
    if missing and var_positional is not None:
        for name in missing:
            arguments.append(deadline if name == "deadline" else seed)
        missing = []
    if missing:
        raise EvalError("solve() must accept both deadline and seed")

    try:
        signature.bind(*arguments, **keywords)
    except _TYPE_ERROR as exc:
        raise EvalError(f"solve() signature cannot accept the evaluator contract: {exc}") from exc
    try:
        return solver(*arguments, **keywords)
    except _EXCEPTION as exc:
        raise EvalError(f"solve() raised: {exc}") from exc


def _load_candidate_result(
    candidate_dir: Path,
    cell: CellSpec,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise EvalError("candidate is missing solver.py")
    module_name = f"_autoevolve_matmul_{_ABS(_HASH(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load candidate solver.py")
    module = importlib.util.module_from_spec(spec)
    _MODULES[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except _EXCEPTION as exc:
            raise EvalError(f"solver.py failed to import: {exc}") from exc
        solver = _GETATTR(module, "solve", None)
        if not _CALLABLE(solver):
            raise EvalError("solver.py must define callable solve()")
        raw = _call_solver(solver, _candidate_payload(cell), deadline, cell.seed)
        return _normalize_decomposition(raw, cell)
    finally:
        _DICT_POP(_MODULES, module_name, None)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Search once, snapshot once, reconstruct the full tensor, and score rank."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    cell = CELL
    candidate_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if candidate_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")
    deadline = _MONOTONIC() + candidate_budget
    matrices = _load_candidate_result(candidate_dir, cell, deadline)
    rank = matrices[0].shape[0]
    if cell.coefficient_mode == "exact":
        numeric_tolerance = _verify_exact(matrices, cell)
    else:
        numeric_tolerance = _verify_numeric(matrices, cell)
    sparsity, distinct_values, coefficient_entries = _descriptor_metrics(matrices, cell)
    return {
        GATE: 1.0,
        METRIC: _FLOAT(rank),
        "target_rank": _FLOAT(cell.target_rank),
        "m": _FLOAT(cell.m),
        "k": _FLOAT(cell.k),
        "n": _FLOAT(cell.n),
        "coefficient_sparsity": sparsity,
        "distinct_coefficient_values": distinct_values,
        "coefficient_entries": coefficient_entries,
        "numeric_tolerance": numeric_tolerance,
        "stage_reached": _FLOAT(stage),
    }


def ceiling() -> dict[str, float | str] | None:
    """No evaluator-derived global rank lower bound is asserted here."""

    return None
