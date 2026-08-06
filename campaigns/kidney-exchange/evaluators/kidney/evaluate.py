"""Exact gate for generated kidney exchange cycle and chain packings."""

from __future__ import annotations

import importlib.util
import inspect
import operator
import os
import random
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

GATE = "matching_valid"
METRIC = "transplants"
MAXIMIZE = True

DESCRIPTORS = [
    {
        "name": "chain_share",
        "metric": "chain_share",
        "bins": 10,
        "lo": 0.0,
        "hi": 1.0,
    },
    {
        "name": "mean_cycle_length",
        "metric": "mean_cycle_length",
        "bins": 8,
        "lo": 0.0,
        "hi": 4.0,
    },
]

_BLOOD_WEIGHTS = (("O", 45), ("A", 40), ("B", 11), ("AB", 4))
_PRA_WEIGHTS = (("low", 70), ("medium", 20), ("high", 10))
_POSITIVE_CROSSMATCH_PERCENT = {"low": 5, "medium": 45, "high": 90}
_DEADLINE_HEADROOM_S = 3.0

_PROTECTED_REPORT_NAMES = frozenset(
    {
        GATE,
        METRIC,
        "baseline_transplants",
        "baseline_cycle_count",
        "baseline_chain_count",
        "baseline_time_ms",
        "cycle_count",
        "chain_count",
        "cycle_transplants",
        "chain_transplants",
        "chain_share",
        "mean_cycle_length",
        "pair_count",
        "altruist_count",
        "stage_reached",
    }
)

_BOOL = bool
_CALLABLE = callable
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_INDEX = operator.index
_INT = int
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_LIST = list
_MONOTONIC = time.monotonic
_NEXT = next
_NP_BOOL_TYPE = type(np.bool_(False))
_PERF_COUNTER_NS = time.perf_counter_ns
_RANGE = range
_SET = set
_SIGNATURE = inspect.signature
_SORTED = sorted
_STOP_ITERATION = StopIteration
_STR = str
_TEXT_TYPES = (str, bytes, bytearray)
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_VARS = vars
_ANY = any


@dataclass(frozen=True)
class CellSpec:
    key: str
    seed: int
    pair_count: int
    altruist_count: int
    cycle_cap: int
    chain_cap: int
    timeout_s: float
    require_validation_shape: bool = False


@dataclass(frozen=True)
class PairProfile:
    patient_blood: str
    donor_blood: str
    pra_tier: str


@dataclass(frozen=True)
class Instance:
    cell: CellSpec
    generation_seed: int
    pairs: tuple[PairProfile, ...]
    altruist_blood: tuple[str, ...]
    adjacency: tuple[tuple[int, ...], ...]

    @property
    def altruists(self) -> tuple[int, ...]:
        start = self.cell.pair_count
        return _TUPLE(_RANGE(start, start + self.cell.altruist_count))

    @property
    def vertex_count(self) -> int:
        return self.cell.pair_count + self.cell.altruist_count


@dataclass(frozen=True)
class Solution:
    cycles: tuple[tuple[int, ...], ...]
    chains: tuple[tuple[int, ...], ...]

    def wire(self) -> dict[str, object]:
        return {
            "cycles": [list(cycle) for cycle in self.cycles],
            "chains": [list(chain) for chain in self.chains],
        }


@dataclass(frozen=True)
class Measurement:
    transplants: int
    cycle_count: int
    chain_count: int
    cycle_transplants: int
    chain_transplants: int


@dataclass(frozen=True)
class _Option:
    kind: str
    vertices: tuple[int, ...]
    mask: int
    transplants: int


_CELLS = {
    "small-validation": CellSpec(
        key="small-validation",
        seed=81021,
        pair_count=8,
        altruist_count=1,
        cycle_cap=3,
        chain_cap=4,
        timeout_s=15.0,
        require_validation_shape=True,
    ),
    "pairs-80-frontier": CellSpec(
        key="pairs-80-frontier",
        seed=81022,
        pair_count=80,
        altruist_count=2,
        cycle_cap=3,
        chain_cap=6,
        timeout_s=45.0,
    ),
    "pairs-160-frontier": CellSpec(
        key="pairs-160-frontier",
        seed=81023,
        pair_count=160,
        altruist_count=4,
        cycle_cap=3,
        chain_cap=8,
        timeout_s=60.0,
    ),
}


def _draw_weighted(rng: random.Random, choices: tuple[tuple[str, int], ...]) -> str:
    total = sum(weight for _, weight in choices)
    draw = rng.randrange(total)
    for value, weight in choices:
        if draw < weight:
            return value
        draw -= weight
    raise EvalError("weighted draw exhausted its exact integer distribution")


def _abo_compatible(donor: str, patient: str) -> bool:
    if donor == "O":
        return True
    if donor == "A":
        return patient in {"A", "AB"}
    if donor == "B":
        return patient in {"B", "AB"}
    return donor == "AB" and patient == "AB"


def _sample_incompatible_pair(rng: random.Random) -> PairProfile:
    for _ in range(10_000):
        patient_blood = _draw_weighted(rng, _BLOOD_WEIGHTS)
        donor_blood = _draw_weighted(rng, _BLOOD_WEIGHTS)
        pra_tier = _draw_weighted(rng, _PRA_WEIGHTS)
        blood_compatible = _abo_compatible(donor_blood, patient_blood)
        crossmatch_positive = (
            rng.randrange(100) < _POSITIVE_CROSSMATCH_PERCENT[pra_tier]
        )
        if not blood_compatible or crossmatch_positive:
            return PairProfile(patient_blood, donor_blood, pra_tier)
    raise EvalError("could not generate an incompatible pair within the operation cap")


def _build_adjacency(
    spec: CellSpec,
    pairs: tuple[PairProfile, ...],
    altruist_blood: tuple[str, ...],
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed ^ 0x4B49_444E_4559)
    donor_blood = tuple(pair.donor_blood for pair in pairs) + altruist_blood
    rows: list[tuple[int, ...]] = []
    for donor_vertex, blood in enumerate(donor_blood):
        targets: list[int] = []
        for patient_vertex, pair in enumerate(pairs):
            if donor_vertex == patient_vertex:
                continue
            if not _abo_compatible(blood, pair.patient_blood):
                continue
            if rng.randrange(100) < _POSITIVE_CROSSMATCH_PERCENT[pair.pra_tier]:
                continue
            targets.append(patient_vertex)
        rows.append(tuple(targets))
    if len(rows) != spec.pair_count + spec.altruist_count:
        raise EvalError("generated graph has the wrong number of donor vertices")
    return tuple(rows)


def _find_first_cycle(
    instance: Instance,
    blocked: set[int],
    length: int,
) -> tuple[int, ...] | None:
    pair_count = instance.cell.pair_count

    def extend(path: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(path) == length:
            return path if path[0] in instance.adjacency[path[-1]] else None
        for target in instance.adjacency[path[-1]]:
            if target >= pair_count or target in blocked or target in path:
                continue
            found = extend((*path, target))
            if found is not None:
                return found
        return None

    for start in range(pair_count):
        if start in blocked:
            continue
        found = extend((start,))
        if found is not None:
            return found
    return None


def _has_missing_pair_edge(instance: Instance) -> bool:
    for left in range(instance.cell.pair_count):
        for right in range(left + 1, instance.cell.pair_count):
            if right not in instance.adjacency[left] or left not in instance.adjacency[right]:
                return True
    return False


def _validation_shape_is_useful(instance: Instance) -> bool:
    has_cycle = any(
        _find_first_cycle(instance, set(), length) is not None
        for length in range(2, instance.cell.cycle_cap + 1)
    )
    has_altruist_edge = any(instance.adjacency[altruist] for altruist in instance.altruists)
    return has_cycle and has_altruist_edge and _has_missing_pair_edge(instance)


def _generate_instance(spec: CellSpec) -> Instance:
    for attempt in range(1_000):
        seed = spec.seed + attempt
        rng = random.Random(seed)
        pairs = tuple(_sample_incompatible_pair(rng) for _ in range(spec.pair_count))
        altruist_blood = tuple(
            _draw_weighted(rng, _BLOOD_WEIGHTS) for _ in range(spec.altruist_count)
        )
        instance = Instance(
            cell=spec,
            generation_seed=seed,
            pairs=pairs,
            altruist_blood=altruist_blood,
            adjacency=_build_adjacency(spec, pairs, altruist_blood, seed),
        )
        if not spec.require_validation_shape or _validation_shape_is_useful(instance):
            return instance
    raise EvalError("validation graph generation exhausted its deterministic attempt cap")


_CELL_KEY = os.environ.get("AUTOEVOLVE_CELL", "small-validation")
if _CELL_KEY not in _CELLS:
    _choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL_KEY!r}")
CELL = _CELLS[_CELL_KEY]
INSTANCE = _generate_instance(CELL)
STAGES: list[StageSpec] = [
    StageSpec(name="generated-instance-search-and-exact-gate", timeout_s=CELL.timeout_s),
]


def _snapshot_sequence(raw: object, field: str, limit: int) -> tuple[object, ...]:
    if _ISINSTANCE(raw, _TEXT_TYPES) or _ISINSTANCE(raw, Mapping):
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
    if not _ISINSTANCE(raw, Mapping):
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
    return snapshot


def _exact_int(raw: object, field: str) -> int:
    if _TYPE(raw) in {_BOOL, _NP_BOOL_TYPE}:
        raise EvalError(f"{field} must be an integer, got bool")
    try:
        return _INT(_INDEX(raw))
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be an integer, got {_TYPE(raw).__name__}") from exc


def _normalize_routes(
    raw: object,
    field: str,
    instance: Instance,
) -> tuple[tuple[int, ...], ...]:
    groups = _snapshot_sequence(raw, field, instance.vertex_count)
    normalized: list[tuple[int, ...]] = []
    for group_index, group in _ENUMERATE(groups):
        values = _snapshot_sequence(
            group,
            f"{field}[{group_index}]",
            instance.vertex_count,
        )
        normalized.append(
            _TUPLE(
                _exact_int(value, f"{field}[{group_index}][{value_index}]")
                for value_index, value in _ENUMERATE(values)
            )
        )
    return _TUPLE(normalized)


def _normalize_solution(raw: object, instance: Instance = INSTANCE) -> Solution:
    """Consume candidate containers once and retain immutable primitive values."""

    values = _mapping_snapshot(raw)
    expected = {"cycles", "chains"}
    found = _SET(values)
    if found != expected:
        missing = _SORTED(expected - found)
        extra = _SORTED(found - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise EvalError(f"solution schema is exact; {'; '.join(details)}")
    return Solution(
        cycles=_normalize_routes(values["cycles"], "cycles", instance),
        chains=_normalize_routes(values["chains"], "chains", instance),
    )


def _check_unused(vertices: tuple[int, ...], used: set[int], field: str) -> None:
    local: set[int] = _SET()
    for vertex in vertices:
        if vertex in local:
            raise EvalError(f"{field} repeats vertex {vertex}")
        if vertex in used:
            raise EvalError(f"vertex {vertex} is used more than once across the solution")
        local.add(vertex)
    used.update(local)


def _verify_solution(solution: Solution, instance: Instance = INSTANCE) -> Measurement:
    used: set[int] = _SET()
    pair_count = instance.cell.pair_count
    altruists = _SET(instance.altruists)

    for cycle_index, cycle in _ENUMERATE(solution.cycles):
        if _LEN(cycle) < 2:
            raise EvalError(f"cycle {cycle_index} must contain at least two pair vertices")
        if _LEN(cycle) > instance.cell.cycle_cap:
            raise EvalError(
                f"cycle {cycle_index} length {_LEN(cycle)} exceeds cycle cap "
                f"{instance.cell.cycle_cap}"
            )
        for vertex in cycle:
            if not 0 <= vertex < pair_count:
                raise EvalError(f"cycle {cycle_index} vertex {vertex} is not a pair vertex")
        _check_unused(cycle, used, f"cycle {cycle_index}")

    for chain_index, chain in _ENUMERATE(solution.chains):
        if _LEN(chain) < 2:
            raise EvalError(
                f"chain {chain_index} must contain an altruist and at least one pair"
            )
        edge_count = _LEN(chain) - 1
        if edge_count > instance.cell.chain_cap:
            raise EvalError(
                f"chain {chain_index} length {edge_count} exceeds chain cap "
                f"{instance.cell.chain_cap}"
            )
        if chain[0] not in altruists:
            raise EvalError(f"chain {chain_index} must start at an altruistic donor")
        for vertex in chain[1:]:
            if not 0 <= vertex < pair_count:
                raise EvalError(f"chain {chain_index} vertex {vertex} is not a pair vertex")
        _check_unused(chain, used, f"chain {chain_index}")

    cycle_transplants = 0
    for cycle_index, cycle in _ENUMERATE(solution.cycles):
        for edge_index, donor in _ENUMERATE(cycle):
            patient = cycle[(edge_index + 1) % _LEN(cycle)]
            if patient not in instance.adjacency[donor]:
                raise EvalError(
                    f"cycle {cycle_index} edge {donor}->{patient} does not exist"
                )
        cycle_transplants += _LEN(cycle)

    chain_transplants = 0
    for chain_index, chain in _ENUMERATE(solution.chains):
        edge_count = _LEN(chain) - 1
        for edge_index in _RANGE(edge_count):
            donor = chain[edge_index]
            patient = chain[edge_index + 1]
            if patient not in instance.adjacency[donor]:
                raise EvalError(
                    f"chain {chain_index} edge {donor}->{patient} does not exist"
                )
        chain_transplants += edge_count

    return Measurement(
        transplants=cycle_transplants + chain_transplants,
        cycle_count=_LEN(solution.cycles),
        chain_count=_LEN(solution.chains),
        cycle_transplants=cycle_transplants,
        chain_transplants=chain_transplants,
    )


def _reject_metric_names(module: ModuleType) -> None:
    claimed = _SORTED(_PROTECTED_REPORT_NAMES.intersection(_VARS(module)))
    if claimed:
        raise EvalError(f"candidate declared self-reported metric names: {', '.join(claimed)}")


def _candidate_payload(instance: Instance) -> dict[str, object]:
    return {
        "cell": instance.cell.key,
        "seed": instance.generation_seed,
        "pair_count": instance.cell.pair_count,
        "altruists": instance.altruists,
        "cycle_cap": instance.cell.cycle_cap,
        "chain_cap": instance.cell.chain_cap,
        "edges": instance.adjacency,
    }


def _call_solver(
    solver: Callable[..., object],
    payload: dict[str, object],
    deadline: float,
) -> object:
    try:
        parameters = _LIST(_SIGNATURE(solver).parameters.values())
    except (_TYPE_ERROR, _VALUE_ERROR) as exc:
        raise EvalError(f"could not inspect solve() signature: {exc}") from exc
    accepts_instance = _ANY(
        parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        for parameter in parameters
    )
    if not accepts_instance:
        raise EvalError("solve() must accept the generated instance as a positional argument")

    args: list[object] = [payload]
    kwargs: dict[str, object] = {}
    deadline_parameter = _NEXT(
        (parameter for parameter in parameters if parameter.name == "deadline"),
        None,
    )
    accepts_var_positional = _ANY(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    accepts_var_keyword = _ANY(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if deadline_parameter is not None:
        if deadline_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(deadline)
        else:
            kwargs["deadline"] = deadline
    elif accepts_var_positional:
        args.append(deadline)
    elif accepts_var_keyword:
        kwargs["deadline"] = deadline
    try:
        return solver(*args, **kwargs)
    except _EXCEPTION as exc:
        raise EvalError(f"solve() raised: {exc}") from exc


def _load_candidate_solution(candidate_dir: Path, deadline: float) -> Solution:
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise EvalError("candidate is missing solver.py")
    module_name = f"_autoevolve_kidney_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load candidate solver.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except _EXCEPTION as exc:
            raise EvalError(f"solver.py failed to import: {exc}") from exc
        _reject_metric_names(module)
        solver = _GETATTR(module, "solve", None)
        if not _CALLABLE(solver):
            raise EvalError("solver.py must define callable solve()")
        raw = _call_solver(solver, _candidate_payload(INSTANCE), deadline)
        _reject_metric_names(module)
        solution = _normalize_solution(raw)
        _reject_metric_names(module)
        return solution
    finally:
        sys.modules.pop(module_name, None)


def _greedy_solution(instance: Instance) -> Solution:
    used: set[int] = _SET()
    cycles: list[tuple[int, ...]] = []
    while True:
        chosen = None
        for length in range(2, instance.cell.cycle_cap + 1):
            chosen = _find_first_cycle(instance, used, length)
            if chosen is not None:
                break
        if chosen is None:
            break
        cycles.append(chosen)
        used.update(chosen)
    return Solution(cycles=tuple(cycles), chains=())


def _enumerate_cycles(instance: Instance) -> tuple[tuple[int, ...], ...]:
    found: set[tuple[int, ...]] = _SET()
    pair_count = instance.cell.pair_count

    def extend(start: int, path: tuple[int, ...], target_length: int) -> None:
        if _LEN(path) == target_length:
            if start in instance.adjacency[path[-1]]:
                found.add(path)
            return
        for target in instance.adjacency[path[-1]]:
            if target >= pair_count or target < start or target in path:
                continue
            extend(start, (*path, target), target_length)

    for length in range(2, instance.cell.cycle_cap + 1):
        for start in range(pair_count):
            extend(start, (start,), length)
    return tuple(sorted(found, key=lambda cycle: (len(cycle), cycle)))


def _enumerate_chains(instance: Instance) -> tuple[tuple[int, ...], ...]:
    found: list[tuple[int, ...]] = []
    pair_count = instance.cell.pair_count

    def extend(path: tuple[int, ...]) -> None:
        if _LEN(path) > 1:
            found.append(path)
        if _LEN(path) - 1 == instance.cell.chain_cap:
            return
        for target in instance.adjacency[path[-1]]:
            if target >= pair_count or target in path:
                continue
            extend((*path, target))

    for altruist in instance.altruists:
        extend((altruist,))
    return tuple(found)


def _option(kind: str, vertices: tuple[int, ...]) -> _Option:
    mask = 0
    for vertex in vertices:
        mask |= 1 << vertex
    transplants = _LEN(vertices) if kind == "cycle" else _LEN(vertices) - 1
    return _Option(kind, vertices, mask, transplants)


def _exact_result(instance: Instance) -> tuple[Solution, int]:
    """Solve the validation cell exactly by finite set-packing dynamic programming."""

    if not instance.cell.require_validation_shape:
        raise EvalError("the exact solver is restricted to the small validation cell")
    options = tuple(_option("cycle", cycle) for cycle in _enumerate_cycles(instance)) + tuple(
        _option("chain", chain) for chain in _enumerate_chains(instance)
    )
    states: dict[int, tuple[int, tuple[_Option, ...]]] = {0: (0, ())}
    for option in options:
        prior = tuple(states.items())
        for used_mask, (score, selected) in prior:
            if used_mask & option.mask:
                continue
            next_mask = used_mask | option.mask
            next_score = score + option.transplants
            incumbent = states.get(next_mask)
            if incumbent is None or next_score > incumbent[0]:
                states[next_mask] = (next_score, (*selected, option))
    optimum, selected = max(states.values(), key=lambda item: item[0])
    cycles = tuple(option.vertices for option in selected if option.kind == "cycle")
    chains = tuple(option.vertices for option in selected if option.kind == "chain")
    return Solution(cycles=cycles, chains=chains), optimum


def exact_validation_solution() -> dict[str, object]:
    """Return the exact small-cell solution for a validation test candidate."""

    solution, _ = _exact_result(INSTANCE)
    return solution.wire()


def exact_validation_optimum() -> int:
    """Return the exact transplant optimum for the generated validation cell."""

    _, optimum = _exact_result(INSTANCE)
    return optimum


def _metrics(
    measured: Measurement,
    baseline: Measurement,
    baseline_ns: int,
    stage: int,
) -> dict[str, float]:
    chain_share = (
        measured.chain_transplants / measured.transplants if measured.transplants else 0.0
    )
    mean_cycle_length = (
        measured.cycle_transplants / measured.cycle_count if measured.cycle_count else 0.0
    )
    return {
        GATE: 1.0,
        METRIC: _FLOAT(measured.transplants),
        "cycle_count": _FLOAT(measured.cycle_count),
        "chain_count": _FLOAT(measured.chain_count),
        "cycle_transplants": _FLOAT(measured.cycle_transplants),
        "chain_transplants": _FLOAT(measured.chain_transplants),
        "chain_share": chain_share,
        "mean_cycle_length": mean_cycle_length,
        "baseline_transplants": _FLOAT(baseline.transplants),
        "baseline_cycle_count": _FLOAT(baseline.cycle_count),
        "baseline_chain_count": _FLOAT(baseline.chain_count),
        "baseline_time_ms": baseline_ns / 1_000_000,
        "pair_count": _FLOAT(INSTANCE.cell.pair_count),
        "altruist_count": _FLOAT(INSTANCE.cell.altruist_count),
        "stage_reached": _FLOAT(stage),
    }


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run the in-process baseline, normalize one candidate result, and gate it."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    started = _MONOTONIC()
    candidate_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if candidate_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")

    baseline_started = _PERF_COUNTER_NS()
    baseline_solution = _greedy_solution(INSTANCE)
    baseline_ns = _PERF_COUNTER_NS() - baseline_started
    baseline = _verify_solution(baseline_solution)

    candidate = _load_candidate_solution(candidate_dir, started + candidate_budget)
    measured = _verify_solution(candidate)
    return _metrics(measured, baseline, baseline_ns, stage)


def ceiling() -> dict[str, float | str]:
    """At most one transplant can enter each paired patient's vertex."""

    return {
        "metric": METRIC,
        "value": float(INSTANCE.cell.pair_count),
        "method": "vertex disjointness permits at most one transplant per patient pair",
    }
