"""Exact shortest-path correctness and same-run query-throughput evaluator."""

from __future__ import annotations

import heapq
import importlib.util
import inspect
import operator
import os
import random
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [
    StageSpec(name="exact-gate-and-query-timing", timeout_s=240.0, mem_mb=4096),
]
GATE = "exact_shortest_paths"
METRIC = "queries_per_second"
MAXIMIZE = True
DESCRIPTORS = SOURCE_DESCRIPTORS

_DEADLINE_HEADROOM_S = 30.0
_ROUTER_MAX_BYTES = 1 << 20
_PROTECTED_REPORT_NAMES = frozenset(
    {
        GATE,
        METRIC,
        "call_diversity",
        "edge_count",
        "mutable_lines",
        "preprocessing_seconds",
        "query_count",
        "query_seconds",
        "query_speedup",
        "reference_queries_per_second",
        "reference_query_seconds",
        "validation_all_pairs",
        "vertex_count",
    }
)

_BOOL = bool
_CALLABLE = callable
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_HEAPPOP = heapq.heappop
_HEAPPUSH = heapq.heappush
_INDEX = operator.index
_INT = int
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_MAX = max
_MONOTONIC = time.monotonic
_NEXT = next
_NP_BOOL_TYPE = type(np.bool_(False))
_PERF_COUNTER_NS = time.perf_counter_ns
_RANGE = range
_SET = set
_SIGNATURE = inspect.signature
_SORTED = sorted
_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_STOP_ITERATION = StopIteration
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_VARS = vars
_ZIP = zip


@dataclass(frozen=True)
class _CellSpec:
    key: str
    width: int
    height: int
    graph_seed: int
    query_seed: int
    query_count: int
    exhaustive: bool


class _Graph(NamedTuple):
    vertex_count: int
    edges: tuple[tuple[int, int, int], ...]
    adjacency: tuple[tuple[tuple[int, int], ...], ...]


class _Answer(NamedTuple):
    distance: int
    path: tuple[int, ...]


_CELLS = {
    "small-validation": _CellSpec(
        key="small-validation",
        width=8,
        height=8,
        graph_seed=2026080601,
        query_seed=2026080602,
        query_count=0,
        exhaustive=True,
    ),
    "large-frontier": _CellSpec(
        key="large-frontier",
        width=72,
        height=72,
        graph_seed=2026080611,
        query_seed=2026080612,
        query_count=512,
        exhaustive=False,
    ),
}
_CELL_KEY = os.environ.get("AUTOEVOLVE_CELL", "small-validation")
if _CELL_KEY not in _CELLS:
    _choices = ", ".join(_SORTED(_CELLS))
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL_KEY!r}")
CELL = _CELLS[_CELL_KEY]


def _add_edge(roads: list[dict[int, int]], source: int, target: int, weight: int) -> None:
    """Keep one positive minimum-weight directed road for each endpoint pair."""

    current = roads[source].get(target)
    if current is None or weight < current:
        roads[source][target] = weight


def _generate_graph(cell: _CellSpec) -> _Graph:
    """Build a deterministic grid, connector, and arterial road model."""

    rng = random.Random(cell.graph_seed)
    vertex_count = cell.width * cell.height
    roads: list[dict[int, int]] = [{} for _ in _RANGE(vertex_count)]

    def vertex(x: int, y: int) -> int:
        return y * cell.width + x

    for y in _RANGE(cell.height):
        for x in _RANGE(cell.width):
            here = vertex(x, y)
            if x + 1 < cell.width:
                right = vertex(x + 1, y)
                _add_edge(roads, here, right, 90 + rng.randrange(31))
                _add_edge(roads, right, here, 90 + rng.randrange(31))
            if y + 1 < cell.height:
                below = vertex(x, y + 1)
                _add_edge(roads, here, below, 90 + rng.randrange(31))
                _add_edge(roads, below, here, 90 + rng.randrange(31))
            if x + 1 < cell.width and y + 1 < cell.height and rng.randrange(5) == 0:
                diagonal = vertex(x + 1, y + 1)
                _add_edge(roads, here, diagonal, 125 + rng.randrange(31))
                _add_edge(roads, diagonal, here, 125 + rng.randrange(31))

    stride = 4
    for y in _RANGE(0, cell.height, 8):
        for x in _RANGE(0, cell.width - stride, stride):
            left = vertex(x, y)
            right = vertex(x + stride, y)
            _add_edge(roads, left, right, 285 + rng.randrange(31))
            _add_edge(roads, right, left, 285 + rng.randrange(31))
    for x in _RANGE(0, cell.width, 8):
        for y in _RANGE(0, cell.height - stride, stride):
            top = vertex(x, y)
            bottom = vertex(x, y + stride)
            _add_edge(roads, top, bottom, 285 + rng.randrange(31))
            _add_edge(roads, bottom, top, 285 + rng.randrange(31))

    adjacency = _TUPLE(
        _TUPLE(_SORTED(neighbors.items()))
        for neighbors in roads
    )
    edges = _TUPLE(
        (source, target, weight)
        for source, neighbors in _ENUMERATE(adjacency)
        for target, weight in neighbors
    )
    return _Graph(vertex_count, edges, adjacency)


def _generate_queries(
    cell: _CellSpec,
    vertex_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return all validation pairs or one fixed seeded frontier query set."""

    if cell.exhaustive:
        return _TUPLE(
            (source, target)
            for source in _RANGE(vertex_count)
            for target in _RANGE(vertex_count)
            if source != target
        )

    rng = random.Random(cell.query_seed)
    queries: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = _SET()
    minimum_grid_distance = (cell.width + cell.height) // 3
    while _LEN(queries) < cell.query_count:
        source = rng.randrange(vertex_count)
        target = rng.randrange(vertex_count)
        if source == target:
            continue
        source_x, source_y = source % cell.width, source // cell.width
        target_x, target_y = target % cell.width, target // cell.width
        grid_distance = abs(source_x - target_x) + abs(source_y - target_y)
        pair = (source, target)
        if grid_distance < minimum_grid_distance or pair in seen:
            continue
        seen.add(pair)
        queries.append(pair)
    return _TUPLE(queries)


def _plain_dijkstra(
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    source: int,
    target: int,
) -> _Answer:
    """Run textbook binary-heap Dijkstra and reconstruct one shortest path."""

    distances: list[int | None] = [None] * _LEN(adjacency)
    parents = [-1] * _LEN(adjacency)
    distances[source] = 0
    heap: list[tuple[int, int]] = [(0, source)]
    while heap:
        distance, node = _HEAPPOP(heap)
        if distances[node] != distance:
            continue
        if node == target:
            break
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            known = distances[neighbor]
            if known is None or candidate < known:
                distances[neighbor] = candidate
                parents[neighbor] = node
                _HEAPPUSH(heap, (candidate, neighbor))

    distance = distances[target]
    if distance is None:
        raise EvalError(f"reference Dijkstra found no path from {source} to {target}")
    reversed_path = [target]
    node = target
    while node != source:
        node = parents[node]
        if node < 0:
            raise EvalError(f"reference path reconstruction failed for {source}->{target}")
        reversed_path.append(node)
    reversed_path.reverse()
    return _Answer(distance=distance, path=_TUPLE(reversed_path))


def _snapshot_iterable(raw: object, label: str, limit: int) -> tuple[object, ...]:
    """Consume one candidate-controlled iterable once with a strict size cap."""

    try:
        iterator: Iterator[object] = _ITER(raw)  # type: ignore[arg-type]
    except _TYPE_ERROR as exc:
        raise EvalError(f"{label} must be iterable, got {_TYPE(raw).__name__}") from exc
    items: list[object] = []
    for index in _RANGE(limit + 1):
        try:
            item = _NEXT(iterator)
        except _STOP_ITERATION:
            return _TUPLE(items)
        except _EXCEPTION as exc:
            raise EvalError(f"{label} failed while reading item {index}: {exc}") from exc
        if index == limit:
            raise EvalError(f"{label} contains more than {limit} items")
        items.append(item)
    raise EvalError(f"{label} could not be normalized")


def _exact_int(raw: object, label: str) -> int:
    """Accept index-compatible integers while rejecting boolean subclasses."""

    if _ISINSTANCE(raw, _BOOL | _NP_BOOL_TYPE):
        raise EvalError(f"{label} must be an integer, got bool")
    try:
        return _INT(_INDEX(raw))
    except _TYPE_ERROR as exc:
        raise EvalError(f"{label} must be an integer, got {_TYPE(raw).__name__}") from exc


def _normalize_answer(raw: object, label: str, vertex_count: int) -> _Answer:
    """Read distance and path exactly once into evaluator-owned primitives."""

    pair = _snapshot_iterable(raw, f"{label} result", 2)
    if _LEN(pair) != 2:
        raise EvalError(f"{label} result must contain exactly distance and path")
    distance = _exact_int(pair[0], f"{label} distance")
    raw_path = _snapshot_iterable(pair[1], f"{label} path", vertex_count)
    path: list[int] = []
    for index, raw_vertex in _ENUMERATE(raw_path):
        vertex = _exact_int(raw_vertex, f"{label} path[{index}]")
        if not 0 <= vertex < vertex_count:
            raise EvalError(
                f"{label} path[{index}]={vertex} is outside 0..{vertex_count - 1}"
            )
        path.append(vertex)
    return _Answer(distance=distance, path=_TUPLE(path))


def _edge_weight(
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    source: int,
    target: int,
) -> int | None:
    for neighbor, weight in adjacency[source]:
        if neighbor == target:
            return weight
    return None


def _validate_answer(
    answer: _Answer,
    reference: _Answer,
    query: tuple[int, int],
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    label: str,
) -> None:
    """Apply every exact gate clause to one immutable answer snapshot."""

    source, target = query
    if answer.distance < reference.distance:
        raise EvalError(
            f"{label} returned shorter-than-possible distance {answer.distance}; "
            f"reference is {reference.distance}"
        )
    if answer.distance != reference.distance:
        raise EvalError(
            f"{label} returned distance {answer.distance}; reference is {reference.distance}"
        )
    if not answer.path:
        raise EvalError(f"{label} returned an empty path")
    if answer.path[0] != source or answer.path[-1] != target:
        raise EvalError(
            f"{label} path endpoints are {answer.path[0]}->{answer.path[-1]}, "
            f"expected {source}->{target}"
        )
    if _LEN(_SET(answer.path)) != _LEN(answer.path):
        raise EvalError(f"{label} path contains a repeated vertex")
    total = 0
    # Pairing a list with its own tail is off by one by construction, so
    # strict=True raised on every path including valid ones. Dropping the last
    # element makes the arguments genuinely equal in length, which is what
    # strict exists to check.
    for left, right in _ZIP(answer.path[:-1], answer.path[1:], strict=True):
        weight = _edge_weight(adjacency, left, right)
        if weight is None:
            raise EvalError(f"{label} path uses missing directed edge {left}->{right}")
        total += weight
    if total != answer.distance:
        raise EvalError(
            f"{label} path weight is {total}, but returned distance is {answer.distance}"
        )


def _read_router_source(candidate_dir: Path) -> bytes:
    path = candidate_dir / "router.py"
    if not path.is_file():
        raise EvalError("candidate is missing router.py")
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"could not read router.py: {exc}") from exc
    if _LEN(source) > _ROUTER_MAX_BYTES:
        raise EvalError(f"router.py is {_LEN(source)} bytes; limit is {_ROUTER_MAX_BYTES}")
    return source


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "router.py"
    module_name = f"_autoevolve_shortest_path_{abs(hash(str(path.resolve())))}"
    spec = _SPEC_FROM_FILE_LOCATION(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load candidate router.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except _EXCEPTION as exc:
            raise EvalError(f"router.py failed to import: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
    claimed = _SORTED(_PROTECTED_REPORT_NAMES.intersection(_VARS(module)))
    if claimed:
        raise EvalError(
            "candidate declared self-reported metric names: " + ", ".join(claimed)
        )
    return module


def _call_builder(
    builder: Callable[..., object],
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float,
) -> object:
    """Pass the deadline only when the inspected signature accepts it."""

    try:
        signature = _SIGNATURE(builder)
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"could not inspect build_router() signature: {exc}") from exc
    try:
        signature.bind(vertex_count, edges, deadline)
    except _TYPE_ERROR:
        try:
            signature.bind(vertex_count, edges, deadline=deadline)
        except _TYPE_ERROR:
            try:
                signature.bind(vertex_count, edges)
            except _TYPE_ERROR as exc:
                raise EvalError(
                    "build_router() must accept vertex_count, edges, and optional deadline"
                ) from exc
            return builder(vertex_count, edges)
        return builder(vertex_count, edges, deadline=deadline)
    return builder(vertex_count, edges, deadline)


def _candidate_query(router: object) -> Callable[[int, int], object]:
    query = _GETATTR(router, "query", None)
    if not _CALLABLE(query):
        raise EvalError("build_router() must return an object with callable query()")
    return query


def _seconds(elapsed_ns: int) -> float:
    return _FLOAT(_MAX(1, elapsed_ns)) / 1_000_000_000.0


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Measure exact query throughput after same-run plain-Dijkstra references."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    evaluation_started = _MONOTONIC()
    graph = _generate_graph(CELL)
    queries = _generate_queries(CELL, graph.vertex_count)
    _read_router_source(candidate_dir)
    descriptors = source_metrics(candidate_dir, "router.py")

    reference_started_ns = _PERF_COUNTER_NS()
    references = _TUPLE(
        _plain_dijkstra(graph.adjacency, source, target)
        for source, target in queries
    )
    reference_elapsed_ns = _PERF_COUNTER_NS() - reference_started_ns
    for index, (query, reference) in _ENUMERATE(
        _ZIP(queries, references, strict=True)
    ):
        _validate_answer(
            reference,
            reference,
            query,
            graph.adjacency,
            f"reference query {index} {query[0]}->{query[1]}",
        )

    deadline_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if deadline_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")
    deadline = evaluation_started + deadline_budget
    if _MONOTONIC() >= deadline:
        raise EvalError("reference timing exhausted the candidate preprocessing deadline")

    candidate = _load_candidate(candidate_dir)
    builder = _GETATTR(candidate, "build_router", None)
    if not _CALLABLE(builder):
        raise EvalError("router.py must define callable build_router()")
    preprocessing_started_ns = _PERF_COUNTER_NS()
    try:
        router = _call_builder(builder, graph.vertex_count, graph.edges, deadline)
    except EvalError:
        raise
    except _EXCEPTION as exc:
        raise EvalError(f"build_router() raised: {exc}") from exc
    preprocessing_elapsed_ns = _PERF_COUNTER_NS() - preprocessing_started_ns
    query = _candidate_query(router)

    candidate_started_ns = _PERF_COUNTER_NS()
    answers: list[_Answer] = []
    for index, (source, target) in _ENUMERATE(queries):
        label = f"query {index} {source}->{target}"
        try:
            raw = query(source, target)
        except _EXCEPTION as exc:
            raise EvalError(f"{label} raised: {exc}") from exc
        answers.append(_normalize_answer(raw, label, graph.vertex_count))
    candidate_elapsed_ns = _PERF_COUNTER_NS() - candidate_started_ns

    for index, (query_pair, answer, reference) in _ENUMERATE(
        _ZIP(queries, answers, references, strict=True)
    ):
        _validate_answer(
            answer,
            reference,
            query_pair,
            graph.adjacency,
            f"query {index} {query_pair[0]}->{query_pair[1]}",
        )

    query_count = _LEN(queries)
    reference_seconds = _seconds(reference_elapsed_ns)
    candidate_seconds = _seconds(candidate_elapsed_ns)
    preprocessing_seconds = _seconds(preprocessing_elapsed_ns)
    reference_qps = query_count / reference_seconds
    candidate_qps = query_count / candidate_seconds
    return {
        GATE: 1.0,
        METRIC: candidate_qps,
        "preprocessing_seconds": preprocessing_seconds,
        "query_seconds": candidate_seconds,
        "reference_query_seconds": reference_seconds,
        "reference_queries_per_second": reference_qps,
        "query_speedup": candidate_qps / reference_qps,
        "query_count": _FLOAT(query_count),
        "vertex_count": _FLOAT(graph.vertex_count),
        "edge_count": _FLOAT(_LEN(graph.edges)),
        "validation_all_pairs": _FLOAT(CELL.exhaustive),
        **descriptors,
    }


def ceiling() -> None:
    """No static ceiling exists for machine-specific exact query throughput."""

    return None
