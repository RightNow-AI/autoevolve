"""Recall-gated same-run evaluator for approximate nearest-neighbour search."""

from __future__ import annotations

import importlib.util
import inspect
import math
import operator
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import source_metrics

STAGES: list[StageSpec] = [
    StageSpec(name="build-recall-and-throughput", timeout_s=180.0),
]
GATE = "recall_gate"
METRIC = "queries_per_second"
MAXIMIZE = True

DEADLINE_HEADROOM_S = 30.0
_INDEX_MEMORY_CAP = 1 << 40
_INDEX_OBJECT_CAP = 100_000


class CellSpec(NamedTuple):
    """One generated ANN workload and its exact recall requirement."""

    key: str
    role: str
    vector_count: int
    dimensions: int
    query_count: int
    k: int
    clusters: int
    seed: int
    recall_numerator: int
    recall_denominator: int


_CELLS = {
    "tiny-r100-validation": CellSpec(
        "tiny-r100-validation",
        "validation",
        256,
        16,
        24,
        5,
        8,
        1101,
        1,
        1,
    ),
    "medium-r095-frontier": CellSpec(
        "medium-r095-frontier",
        "frontier",
        4096,
        32,
        96,
        10,
        48,
        2202,
        19,
        20,
    ),
    "large-r090-frontier": CellSpec(
        "large-r090-frontier",
        "frontier",
        12000,
        48,
        160,
        10,
        96,
        3303,
        9,
        10,
    ),
}

_CELL_KEY = os.environ.get("AUTOEVOLVE_CELL", "tiny-r100-validation")
if _CELL_KEY not in _CELLS:
    _choices = ", ".join(sorted(_CELLS))
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL_KEY!r}")
CELL = _CELLS[_CELL_KEY]

_PROTECTED_REPORT_NAMES = frozenset(
    {
        GATE,
        METRIC,
        "recall_at_k",
        "index_build_seconds",
        "candidate_search_seconds",
        "exact_queries_per_second",
        "exact_search_seconds",
        "index_memory_log2",
        "mutable_lines",
        "call_diversity",
    }
)

# Candidate code runs in this interpreter. Bind the primitives used after import so
# replacing a module attribute cannot change normalization, timing, or gate arithmetic.
_ARRAY = np.array
_ASARRAY = np.asarray
_ARANGE = np.arange
_EINSUM = np.einsum
_LEX_SORT = np.lexsort
_DEFAULT_RNG = np.random.default_rng
_NDARRAY_TYPE = np.ndarray
_ANY = any
_BOOL_TYPE = bool
_CALLABLE = callable
_BYTEARRAY_TYPE = bytearray
_BYTES_TYPE = bytes
_COMPLEX_TYPE = complex
_DICT_TYPE = dict
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_FROZENSET_TYPE = frozenset
_GETATTR = getattr
_GETSIZEOF = sys.getsizeof
_ID = id
_INDEX = operator.index
_INSPECT_SIGNATURE = inspect.signature
_INT = int
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_LIST = list
_LOG2 = math.log2
_MAX = max
_MEMORYVIEW_TYPE = memoryview
_MIN = min
_MONOTONIC = time.monotonic
_NEXT = next
_NONE_TYPE = type(None)
_OBJECT_GETATTRIBUTE = object.__getattribute__
_PARAMETER = inspect.Parameter
_PERF_COUNTER_NS = time.perf_counter_ns
_SET = set
_SORTED = sorted
_STOP_ITERATION = StopIteration
_STR = str
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_VARS = vars
_ZIP = zip


def _make_workload(cell: CellSpec) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered float32 database and query vectors from one fixed seed."""

    rng = _DEFAULT_RNG(cell.seed)
    centers = rng.normal(0.0, 6.0, size=(cell.clusters, cell.dimensions))
    database_clusters = rng.integers(0, cell.clusters, size=cell.vector_count)
    query_clusters = rng.integers(0, cell.clusters, size=cell.query_count)
    database = centers[database_clusters] + rng.normal(
        0.0,
        0.65,
        size=(cell.vector_count, cell.dimensions),
    )
    queries = centers[query_clusters] + rng.normal(
        0.0,
        0.65,
        size=(cell.query_count, cell.dimensions),
    )
    return (
        _ASARRAY(database, dtype=np.float32),
        _ASARRAY(queries, dtype=np.float32),
    )


def _exact_neighbors(
    database: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> tuple[tuple[int, ...], ...]:
    """Compute deterministic exact top-k sets by a full float64 scan and sort."""

    database64 = _ASARRAY(database, dtype=np.float64)
    queries64 = _ASARRAY(queries, dtype=np.float64)
    indexes = _ARANGE(database64.shape[0], dtype=np.int64)
    rows: list[tuple[int, ...]] = []
    for query in queries64:
        delta = database64 - query
        distances = _EINSUM("ij,ij->i", delta, delta)
        order = _LEX_SORT((indexes, distances))
        rows.append(_TUPLE(_INT(value) for value in order[:k]))
    return _TUPLE(rows)


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "index.py"
    if not path.is_file():
        raise EvalError("candidate is missing index.py")
    module_name = f"_autoevolve_ann_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except _EXCEPTION as exc:
        sys.modules.pop(module_name, None)
        raise EvalError(f"candidate import failed: {exc}") from exc
    claimed = _SORTED(
        name
        for name in _VARS(module)
        if name.casefold() in _PROTECTED_REPORT_NAMES
    )
    if claimed:
        raise EvalError(
            "candidate declared self-reported metric names: " + ", ".join(claimed)
        )
    return module


def _call_with_deadline(
    function: Callable[..., object],
    args: tuple[object, ...],
    deadline: float,
    label: str,
) -> object:
    """Call one candidate entrypoint according to its inspected signature."""

    try:
        signature = _INSPECT_SIGNATURE(function)
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"cannot inspect candidate {label} signature: {exc}") from exc

    positional = _LIST(args)
    keywords: dict[str, object] = {}
    deadline_parameter = signature.parameters.get("deadline")
    has_var_keywords = _ANY(
        parameter.kind is _PARAMETER.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    has_var_positional = _ANY(
        parameter.kind is _PARAMETER.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if deadline_parameter is not None:
        if deadline_parameter.kind is _PARAMETER.POSITIONAL_ONLY:
            positional.append(deadline)
        else:
            keywords["deadline"] = deadline
    elif has_var_keywords:
        keywords["deadline"] = deadline
    elif has_var_positional:
        positional.append(deadline)
    try:
        signature.bind(*positional, **keywords)
    except _TYPE_ERROR as exc:
        raise EvalError(f"candidate {label} has incompatible signature: {exc}") from exc
    try:
        return function(*positional, **keywords)
    except _EXCEPTION as exc:
        raise EvalError(f"candidate {label} failed: {exc}") from exc


def _snapshot_iterable(raw: object, field: str) -> tuple[object, ...]:
    """Consume one candidate-controlled iterable exactly once."""

    if _ISINSTANCE(raw, Mapping):
        reported = _SORTED(
            _STR(key)
            for key in raw
            if _STR(key).casefold() in _PROTECTED_REPORT_NAMES
        )
        if reported:
            raise EvalError(
                f"{field} returned self-reported metrics: {', '.join(reported)}"
            )
        raise EvalError(f"{field} must return indices, not a mapping")
    try:
        iterator = _ITER(raw)
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be iterable, got {_TYPE(raw).__name__}") from exc
    items: list[object] = []
    item_index = 0
    while True:
        try:
            item = _NEXT(iterator)
        except _STOP_ITERATION:
            break
        except _EXCEPTION as exc:
            raise EvalError(f"{field} failed while reading item {item_index}: {exc}") from exc
        items.append(item)
        item_index += 1
    return _TUPLE(items)


def _exact_index(raw: object, field: str) -> int:
    if _ISINSTANCE(raw, _BOOL_TYPE):
        raise EvalError(f"{field} must be an integer, got bool")
    try:
        return _INT(_INDEX(raw))
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be an integer, got {_TYPE(raw).__name__}") from exc


def _normalize_results(
    raw: object,
    query_count: int,
    k: int,
    vector_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Snapshot result rows once and validate exact integer index structure."""

    raw_rows = _snapshot_iterable(raw, "search() result")
    if _LEN(raw_rows) != query_count:
        raise EvalError(
            f"search() returned {_LEN(raw_rows)} rows; expected {query_count}"
        )
    normalized: list[tuple[int, ...]] = []
    for query_index, raw_row in _ENUMERATE(raw_rows):
        row_items = _snapshot_iterable(raw_row, f"query {query_index} result")
        if _LEN(row_items) != k:
            raise EvalError(
                f"query {query_index} returned {_LEN(row_items)} indices; expected {k}"
            )
        row: list[int] = []
        seen: set[int] = _SET()
        for result_index, raw_index in _ENUMERATE(row_items):
            value = _exact_index(
                raw_index,
                f"query {query_index} index {result_index}",
            )
            if value < 0 or value >= vector_count:
                raise EvalError(
                    f"query {query_index} returned index {value} out of range "
                    f"0..{vector_count - 1}"
                )
            if value in seen:
                raise EvalError(
                    f"query {query_index} returned duplicate index {value}"
                )
            seen.add(value)
            row.append(value)
        normalized.append(_TUPLE(row))
    return _TUPLE(normalized)


def _index_memory_bytes(index: object) -> int:
    """Estimate bytes reachable from the returned index without custom iteration."""

    seen: set[int] = _SET()

    def visit(value: object) -> int:
        if _LEN(seen) >= _INDEX_OBJECT_CAP:
            return _INDEX_MEMORY_CAP
        identity = _ID(value)
        if identity in seen:
            return 0
        seen.add(identity)
        value_type = _TYPE(value)
        if value_type is _NDARRAY_TYPE:
            return _MAX(_INT(_GETSIZEOF(value)), _INT(value.nbytes) + 128)
        if value_type in {
            _STR,
            _BYTES_TYPE,
            _BYTEARRAY_TYPE,
            _INT,
            _FLOAT,
            _COMPLEX_TYPE,
            _BOOL_TYPE,
            _NONE_TYPE,
        }:
            return _INT(_GETSIZEOF(value))
        if value_type is _DICT_TYPE:
            total = _INT(_GETSIZEOF(value))
            for key, item in value.items():
                total += visit(key) + visit(item)
            return _clamp_memory(total)
        if value_type in {_LIST, _TUPLE, _SET, _FROZENSET_TYPE}:
            total = _INT(_GETSIZEOF(value))
            for item in value:
                total += visit(item)
            return _clamp_memory(total)
        if value_type is _MEMORYVIEW_TYPE:
            return _clamp_memory(_INT(_GETSIZEOF(value)) + _INT(value.nbytes))
        if _ISINSTANCE(value, ModuleType):
            return _INT(_GETSIZEOF(value))
        total = 64
        try:
            attributes = _OBJECT_GETATTRIBUTE(value, "__dict__")
        except _EXCEPTION:
            attributes = None
        if _TYPE(attributes) is _DICT_TYPE:
            total += visit(attributes)
        return _clamp_memory(total)

    return _MAX(1, _clamp_memory(visit(index)))


def _clamp_memory(value: int) -> int:
    """Clamp descriptor work to a finite structural range."""

    return _MIN(value, _INDEX_MEMORY_CAP)


def _recall_hits(
    returned: tuple[tuple[int, ...], ...],
    ground_truth: tuple[tuple[int, ...], ...],
) -> int:
    """Count exact-neighbour intersections from immutable snapshots only."""

    hits = 0
    for returned_row, truth_row in _ZIP(returned, ground_truth, strict=True):
        hits += _LEN(_SET(returned_row).intersection(truth_row))
    return hits


def _stage_deadline(started: float, stage: int) -> float:
    budget = STAGES[stage].timeout_s - DEADLINE_HEADROOM_S
    if budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")
    return started + budget


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate exact recall, then report measured candidate and brute-force throughput."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    evaluation_started = _MONOTONIC()
    deadline = _stage_deadline(evaluation_started, stage)
    database, queries = _make_workload(CELL)

    exact_started_ns = _PERF_COUNTER_NS()
    ground_truth = _exact_neighbors(database, queries, CELL.k)
    exact_elapsed_ns = _PERF_COUNTER_NS() - exact_started_ns
    if exact_elapsed_ns <= 0:
        raise EvalError("exact brute-force timing clock did not advance")
    source_structure = source_metrics(candidate_dir, "index.py")
    mutable_lines = _FLOAT(source_structure["mutable_lines"])
    call_diversity = _FLOAT(source_structure["call_diversity"])

    build_input = _ARRAY(database, copy=True)
    build_started_ns = _PERF_COUNTER_NS()
    candidate = _load_candidate(candidate_dir)
    build = _GETATTR(candidate, "build", None)
    search = _GETATTR(candidate, "search", None)
    if not _CALLABLE(build):
        raise EvalError("candidate must define callable build")
    if not _CALLABLE(search):
        raise EvalError("candidate must define callable search")
    index = _call_with_deadline(build, (build_input,), deadline, "build()")
    build_elapsed_ns = _PERF_COUNTER_NS() - build_started_ns
    if _MONOTONIC() >= deadline:
        raise EvalError("candidate exhausted its deadline during index build")
    index_memory_log2 = _LOG2(_FLOAT(_index_memory_bytes(index)))

    query_input = _ARRAY(queries, copy=True)
    search_started_ns = _PERF_COUNTER_NS()
    raw_results = _call_with_deadline(
        search,
        (index, query_input, CELL.k),
        deadline,
        "search()",
    )
    results = _normalize_results(
        raw_results,
        CELL.query_count,
        CELL.k,
        CELL.vector_count,
    )
    search_elapsed_ns = _PERF_COUNTER_NS() - search_started_ns
    if search_elapsed_ns <= 0:
        raise EvalError("candidate search timing clock did not advance")
    if _MONOTONIC() >= deadline:
        raise EvalError("candidate exhausted its deadline during search")

    hits = _recall_hits(results, ground_truth)
    total = CELL.query_count * CELL.k
    if hits * CELL.recall_denominator < total * CELL.recall_numerator:
        recall = hits / total
        threshold = CELL.recall_numerator / CELL.recall_denominator
        raise EvalError(
            f"recall gate failed: recall@{CELL.k} {recall:.6f} is below "
            f"{threshold:.6f} ({hits} of {total} exact neighbours)"
        )

    exact_seconds = exact_elapsed_ns / 1_000_000_000.0
    build_seconds = build_elapsed_ns / 1_000_000_000.0
    search_seconds = search_elapsed_ns / 1_000_000_000.0
    return {
        GATE: 1.0,
        "recall_at_k": hits / total,
        METRIC: CELL.query_count / search_seconds,
        "index_build_seconds": build_seconds,
        "candidate_search_seconds": search_seconds,
        "exact_queries_per_second": CELL.query_count / exact_seconds,
        "exact_search_seconds": exact_seconds,
        "index_memory_log2": index_memory_log2,
        "mutable_lines": mutable_lines,
        "call_diversity": call_diversity,
    }


def ceiling() -> None:
    """No static throughput ceiling exists for a measured CPU index."""

    return None


DESCRIPTORS = [
    {
        "name": "index_memory_log2",
        "metric": "index_memory_log2",
        "bins": 12,
        "lo": 6.0,
        "hi": 36.0,
    },
    {
        "name": "call_diversity",
        "metric": "call_diversity",
        "bins": 8,
        "lo": 0.0,
        "hi": 48.0,
    },
]
