"""Exact feasibility gate and distance scorer for Solomon-format CVRPTW instances."""

from __future__ import annotations

import importlib.util
import inspect
import math
import operator
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec
from campaigns.vrp.objective import is_better_result

GATE = "routes_feasible"
METRIC = "total_distance"
MAXIMIZE = False
LEXICOGRAPHIC_OBJECTIVE = ("vehicle_count", METRIC)
DISTANCE_CONVENTION = "euclidean_double_sum_round_half_up_2dp"

DESCRIPTORS = [
    {
        "name": "vehicle_count",
        "metric": "vehicle_count",
        "bins": 256,
        "lo": 1.0,
        "hi": 257.0,
    },
    {
        "name": "mean_route_customers",
        "metric": "mean_route_customers",
        "bins": 100,
        "lo": 1.0,
        "hi": 201.0,
    },
]

_DEADLINE_HEADROOM_S = 3.0
_TIME_TOLERANCE = 1e-9
_TINY_SEED = 730_121
_GENERATED_100_SEED = 730_209


@dataclass(frozen=True)
class Stop:
    customer_id: int
    x: float
    y: float
    demand: float
    earliest: float
    latest: float
    service: float


@dataclass(frozen=True)
class Instance:
    name: str
    vehicle_limit: int
    capacity: float
    stops: tuple[Stop, ...]

    @property
    def depot(self) -> Stop:
        return self.stops[0]

    @property
    def customers(self) -> tuple[Stop, ...]:
        return self.stops[1:]

    @property
    def customer_count(self) -> int:
        return len(self.stops) - 1


@dataclass(frozen=True)
class CellSpec:
    key: str
    fixture: str
    timeout_s: float
    seed: int
    generated_customers: int | None = None


@dataclass(frozen=True)
class Solution:
    routes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Measurement:
    vehicle_count: int
    total_distance: float
    mean_route_customers: float

    @property
    def objective(self) -> tuple[int, float]:
        return (self.vehicle_count, self.total_distance)

    def is_better_than(self, other: Measurement) -> bool:
        return is_better_result(self.objective, other.objective)


_CELLS = {
    "tiny-12-validation": CellSpec(
        key="tiny-12-validation",
        fixture="generated-tiny-12.txt",
        timeout_s=20.0,
        seed=_TINY_SEED,
        generated_customers=12,
    ),
    "generated-100": CellSpec(
        key="generated-100",
        fixture="generated-100.txt",
        timeout_s=90.0,
        seed=_GENERATED_100_SEED,
        generated_customers=100,
    ),
    "solomon-c1-frontier": CellSpec(
        "solomon-c1-frontier", "solomon/C101.txt", 300.0, 731_101
    ),
    "solomon-c2-frontier": CellSpec(
        "solomon-c2-frontier", "solomon/C201.txt", 300.0, 731_201
    ),
    "solomon-r1-frontier": CellSpec(
        "solomon-r1-frontier", "solomon/R101.txt", 300.0, 732_101
    ),
    "solomon-r2-frontier": CellSpec(
        "solomon-r2-frontier", "solomon/R201.txt", 300.0, 732_201
    ),
    "solomon-rc1-frontier": CellSpec(
        "solomon-rc1-frontier", "solomon/RC101.txt", 300.0, 733_101
    ),
    "solomon-rc2-frontier": CellSpec(
        "solomon-rc2-frontier", "solomon/RC201.txt", 300.0, 733_201
    ),
    "homberger-c1-frontier": CellSpec(
        "homberger-c1-frontier", "homberger_200/c1_2_1.txt", 300.0, 741_101
    ),
    "homberger-c2-frontier": CellSpec(
        "homberger-c2-frontier", "homberger_200/c2_2_1.txt", 300.0, 741_201
    ),
    "homberger-r1-frontier": CellSpec(
        "homberger-r1-frontier", "homberger_200/r1_2_1.txt", 300.0, 742_101
    ),
    "homberger-r2-frontier": CellSpec(
        "homberger-r2-frontier", "homberger_200/r2_2_1.txt", 300.0, 742_201
    ),
    "homberger-rc1-frontier": CellSpec(
        "homberger-rc1-frontier", "homberger_200/rc1_2_1.txt", 300.0, 743_101
    ),
    "homberger-rc2-frontier": CellSpec(
        "homberger-rc2-frontier", "homberger_200/rc2_2_1.txt", 300.0, 743_201
    ),
}


_FIXTURES_DIR = (Path(__file__).resolve().parent / "fixtures").resolve()
_FILE_CELL_PREFIX = "file:"
_FILE_CELL_DEFAULT_TIMEOUT_S = _CELLS["solomon-c1-frontier"].timeout_s


def _stable_path_seed(relative_path: str) -> int:
    """Derive a process-stable 32-bit FNV-1a seed from a fixture path."""

    value = 2_166_136_261
    for byte in relative_path.encode("utf-8"):
        value ^= byte
        value = (value * 16_777_619) & 0xFFFF_FFFF
    return value or 1


def _resolve_fixture_path(relative_path: str) -> tuple[str, Path]:
    if not relative_path:
        raise EvalError("file: cell fixture path must not be empty")
    portable = PurePosixPath(relative_path.replace("\\", "/"))
    windows = PureWindowsPath(relative_path)
    if portable.is_absolute() or windows.is_absolute() or windows.drive:
        raise EvalError("file: cell fixture path must be relative to fixtures")
    if ".." in portable.parts or ".." in windows.parts:
        raise EvalError("file: cell fixture path must not contain '..'")

    try:
        resolved = (_FIXTURES_DIR / Path(*portable.parts)).resolve()
    except (OSError, RuntimeError) as exc:
        raise EvalError(f"could not resolve file: cell fixture path: {exc}") from exc
    try:
        canonical = resolved.relative_to(_FIXTURES_DIR)
    except ValueError as exc:
        raise EvalError("file: cell fixture path escapes the fixtures directory") from exc
    return canonical.as_posix(), resolved


def _file_cell_timeout() -> float:
    raw = os.environ.get("AUTOEVOLVE_VRP_TIMEOUT_S")
    if raw is None:
        return _FILE_CELL_DEFAULT_TIMEOUT_S
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise EvalError("AUTOEVOLVE_VRP_TIMEOUT_S must be a positive finite number") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise EvalError("AUTOEVOLVE_VRP_TIMEOUT_S must be a positive finite number")
    return timeout_s


def _resolve_cell(key: str) -> CellSpec:
    named = _CELLS.get(key)
    if named is not None:
        return named
    if key.startswith(_FILE_CELL_PREFIX):
        fixture, _ = _resolve_fixture_path(key.removeprefix(_FILE_CELL_PREFIX))
        return CellSpec(
            key=f"{_FILE_CELL_PREFIX}{fixture}",
            fixture=fixture,
            timeout_s=_file_cell_timeout(),
            seed=_stable_path_seed(fixture),
        )
    choices = ", ".join(_CELLS)
    raise EvalError(
        f"AUTOEVOLVE_CELL must be one of {choices} or file:<fixture path>; got {key!r}"
    )


_CELL_KEY = os.environ.get("AUTOEVOLVE_CELL", "tiny-12-validation")
CELL = _resolve_cell(_CELL_KEY)
STAGES: list[StageSpec] = [
    StageSpec(name="candidate-search-and-exact-vrptw-gate", timeout_s=CELL.timeout_s),
]

_PROTECTED_REPORT_NAMES = frozenset(
    {
        GATE,
        METRIC,
        "vehicle_count",
        "mean_route_customers",
        "customer_count",
        "stage_reached",
    }
)


def _finite_number(token: str, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise EvalError(f"{field} must be numeric, got {token!r}") from exc
    if not math.isfinite(value):
        raise EvalError(f"{field} must be finite")
    return value


def _plain_int(value: float, field: str) -> int:
    integer = int(value)
    if value != integer:
        raise EvalError(f"{field} must be an integer")
    return integer


def parse_solomon_text(text: str, source: str = "<memory>") -> Instance:
    """Parse one standard Solomon or Gehring-Homberger text instance."""

    lines = text.splitlines()
    name = next((line.strip() for line in lines if line.strip()), "")
    if not name:
        raise EvalError(f"{source}: instance name is missing")

    vehicle_marker = next(
        (
            index
            for index, line in enumerate(lines)
            if "NUMBER" in line.upper() and "CAPACITY" in line.upper()
        ),
        None,
    )
    if vehicle_marker is None:
        raise EvalError(f"{source}: NUMBER/CAPACITY header is missing")

    vehicle_values: list[float] | None = None
    for line in lines[vehicle_marker + 1 :]:
        tokens = line.split()
        if len(tokens) != 2:
            continue
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            vehicle_values = values
            break
    if vehicle_values is None:
        raise EvalError(f"{source}: vehicle number and capacity row is missing")
    vehicle_limit = _plain_int(vehicle_values[0], f"{source} vehicle number")
    capacity = vehicle_values[1]
    if vehicle_limit <= 0 or capacity <= 0.0:
        raise EvalError(f"{source}: vehicle number and capacity must be positive")

    customer_marker = next(
        (
            index
            for index, line in enumerate(lines)
            if "CUST" in line.upper()
            and "XCOORD" in line.upper()
            and "YCOORD" in line.upper()
        ),
        None,
    )
    if customer_marker is None:
        raise EvalError(f"{source}: customer column header is missing")

    stops: list[Stop] = []
    for line_number, line in enumerate(lines[customer_marker + 1 :], customer_marker + 2):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 7:
            if any(character.isdigit() for character in line):
                raise EvalError(
                    f"{source}:{line_number}: customer rows must contain seven fields"
                )
            continue
        values = [
            _finite_number(token, f"{source}:{line_number} field {index}")
            for index, token in enumerate(tokens)
        ]
        customer_id = _plain_int(values[0], f"{source}:{line_number} customer id")
        stop = Stop(customer_id, *values[1:])
        if stop.demand < 0.0:
            raise EvalError(f"{source}:{line_number}: demand must be non-negative")
        if stop.earliest > stop.latest:
            raise EvalError(f"{source}:{line_number}: time window is reversed")
        if stop.service < 0.0:
            raise EvalError(f"{source}:{line_number}: service time must be non-negative")
        stops.append(stop)

    if len(stops) < 2:
        raise EvalError(f"{source}: instance must contain a depot and at least one customer")
    ids = [stop.customer_id for stop in stops]
    if len(set(ids)) != len(ids):
        raise EvalError(f"{source}: customer ids must be unique")
    if set(ids) != set(range(len(stops))):
        raise EvalError(f"{source}: customer ids must be contiguous from depot id 0")
    stops.sort(key=lambda stop: stop.customer_id)
    if stops[0].demand != 0.0:
        raise EvalError(f"{source}: depot demand must be zero")
    for stop in stops[1:]:
        if stop.demand > capacity:
            raise EvalError(
                f"{source}: customer {stop.customer_id} demand exceeds vehicle capacity"
            )
    return Instance(name=name, vehicle_limit=vehicle_limit, capacity=capacity, stops=tuple(stops))


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def format_solomon_instance(instance: Instance) -> str:
    """Serialize an instance in the standard seven-column Solomon text layout."""

    rows = [
        instance.name,
        "",
        "VEHICLE",
        "NUMBER     CAPACITY",
        f"{instance.vehicle_limit:6d} {_format_number(instance.capacity):>12}",
        "",
        "CUSTOMER",
        "CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME",
        "",
    ]
    for stop in instance.stops:
        values = (
            str(stop.customer_id),
            _format_number(stop.x),
            _format_number(stop.y),
            _format_number(stop.demand),
            _format_number(stop.earliest),
            _format_number(stop.latest),
            _format_number(stop.service),
        )
        rows.append(" ".join(f"{value:>10}" for value in values))
    return "\n".join(rows) + "\n"


def generate_fixture_text(customer_count: int, seed: int, name: str) -> str:
    """Generate a deterministic, feasible, unpublished Solomon-format fixture."""

    if customer_count < 1:
        raise ValueError("customer_count must be positive")
    state = seed & 0xFFFF_FFFF

    def draw(stop: int) -> int:
        nonlocal state
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        return state % stop

    capacity = 30.0 if customer_count <= 12 else 60.0
    generated: list[Stop] = []
    x = 50.0
    y = 50.0
    clock = 0.0
    customer_rows: list[Stop] = []
    for customer_id in range(1, customer_count + 1):
        next_x = float(5 + draw(91))
        next_y = float(5 + draw(91))
        clock += abs(next_x - x) + abs(next_y - y)
        earliest = float(max(0, int(clock) - draw(16)))
        latest = earliest + float(90 + draw(91))
        service = float(5 + draw(6))
        customer_rows.append(
            Stop(
                customer_id=customer_id,
                x=next_x,
                y=next_y,
                demand=float(1 + draw(9)),
                earliest=earliest,
                latest=latest,
                service=service,
            )
        )
        clock = max(clock, earliest) + service
        x = next_x
        y = next_y
    depot_latest = float(int(clock + abs(x - 50.0) + abs(y - 50.0) + 5_000.0))
    generated.append(Stop(0, 50.0, 50.0, 0.0, 0.0, depot_latest, 0.0))
    generated.extend(customer_rows)
    instance = Instance(
        name=name,
        vehicle_limit=customer_count,
        capacity=capacity,
        stops=tuple(generated),
    )
    return format_solomon_instance(instance)


def _load_instance(cell: CellSpec) -> Instance:
    _, fixture = _resolve_fixture_path(cell.fixture)
    if not fixture.is_file():
        raise EvalError(
            f"cell {cell.key!r} requires fixture {cell.fixture!r}; "
            "run campaigns/vrp/fetch_instances.py through Modal for public instances"
        )
    try:
        text = fixture.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvalError(f"could not read fixture {fixture}: {exc}") from exc
    if cell.generated_customers is not None:
        generated_name = (
            f"AUTOEVOLVE_GENERATED_{cell.generated_customers}_SEED_{cell.seed}"
        )
        expected = generate_fixture_text(cell.generated_customers, cell.seed, generated_name)
        if text != expected:
            raise EvalError(f"generated fixture {cell.fixture!r} does not match its committed seed")
    return parse_solomon_text(text, cell.fixture)


INSTANCE = _load_instance(CELL)

# Candidate import and execution happen in this evaluator process. Bind trusted
# primitives before import so later gate work does not depend on candidate edits.
_ANY = any
_BOOL = bool
_CALLABLE = callable
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_HYPOT = math.hypot
_INDEX = operator.index
_INT = int
_ISINSTANCE = isinstance
_ITER = iter
_LEN = len
_LIST = list
_MAPPING = Mapping
_MAX = max
_MONOTONIC = time.monotonic
_NEXT = next
_NP_BOOL_TYPE = type(np.bool_(False))
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
_VARS = vars


def _round_distance(value: float, floor: Callable[[float], int] = math.floor) -> float:
    return floor(value * 100.0 + 0.5) / 100.0


_ROUND_DISTANCE = _round_distance


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
        items = _snapshot_sequence(raw.items(), "solve() result items", 2)
    except _EXCEPTION as exc:
        if _ISINSTANCE(exc, EvalError):
            raise
        raise EvalError(f"solve() result could not expose items once: {exc}") from exc
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


def _normalize_solution(raw: object, instance: Instance = INSTANCE) -> Solution:
    values = _mapping_snapshot(raw)
    if _SET(values) != {"routes"}:
        missing = _SORTED({"routes"} - _SET(values))
        extra = _SORTED(_SET(values) - {"routes"})
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise EvalError(f"solution schema is exact; {'; '.join(details)}")

    raw_routes = _snapshot_sequence(values["routes"], "routes", instance.vehicle_limit + 1)
    routes: list[tuple[int, ...]] = []
    for route_index, raw_route in _ENUMERATE(raw_routes):
        values = _snapshot_sequence(
            raw_route,
            f"routes[{route_index}]",
            instance.customer_count + 2,
        )
        routes.append(
            _TUPLE(
                _exact_int(value, f"routes[{route_index}][{value_index}]")
                for value_index, value in _ENUMERATE(values)
            )
        )
    return Solution(routes=_TUPLE(routes))


def _distance(
    left: Stop,
    right: Stop,
    hypot: Callable[[float, float], float] = math.hypot,
) -> float:
    return hypot(left.x - right.x, left.y - right.y)


def _verify_solution(
    solution: Solution,
    instance: Instance = INSTANCE,
    *,
    distance: Callable[[Stop, Stop], float] = _distance,
    round_distance: Callable[[float], float] = _round_distance,
    error: type[EvalError] = EvalError,
    measurement_type: type[Measurement] = Measurement,
    tolerance: float = _TIME_TOLERANCE,
    length: Callable[[object], int] = len,
    enumerate_items: Callable[..., object] = enumerate,
    make_set: Callable[..., set[object]] = set,
    make_range: Callable[..., range] = range,
    sort_items: Callable[..., list[object]] = sorted,
    maximum: Callable[..., float] = max,
    stringify: Callable[[object], str] = str,
) -> Measurement:
    if length(solution.routes) > instance.vehicle_limit:
        raise error(
            f"solution uses {length(solution.routes)} vehicles but the instance allows "
            f"{instance.vehicle_limit}"
        )

    stop_by_id = {stop.customer_id: stop for stop in instance.stops}
    required = make_set(make_range(1, instance.customer_count + 1))
    visited: set[int] = make_set()
    total_distance = 0.0

    for route_index, route in enumerate_items(solution.routes):
        if length(route) < 3:
            raise error(f"route {route_index} must contain depot, customer, depot")
        if route[0] != 0 or route[-1] != 0:
            raise error(f"route {route_index} must start and end at depot 0")
        if 0 in route[1:-1]:
            raise error(f"route {route_index} contains depot 0 as a customer")

        demand = 0.0
        clock = instance.depot.earliest + instance.depot.service
        for position, customer_id in enumerate_items(route[1:-1], 1):
            if customer_id not in required:
                raise error(
                    f"route {route_index} position {position} has unknown customer {customer_id}"
                )
            if customer_id in visited:
                raise error(f"customer {customer_id} is visited more than once")
            visited.add(customer_id)
            demand += stop_by_id[customer_id].demand
        if demand > instance.capacity + tolerance:
            raise error(
                f"route {route_index} demand {demand:g} exceeds capacity {instance.capacity:g}"
            )

        for leg_index in make_range(length(route) - 1):
            left_id = route[leg_index]
            right_id = route[leg_index + 1]
            left = stop_by_id[left_id]
            right = stop_by_id[right_id]
            travel = distance(left, right)
            total_distance += travel
            arrival = clock + travel
            service_start = maximum(arrival, right.earliest)
            if service_start > right.latest + tolerance:
                label = "depot" if right_id == 0 else f"customer {right_id}"
                raise error(
                    f"route {route_index} reaches {label} at {service_start:.6f} "
                    f"after window closes at {right.latest:.6f}"
                )
            clock = service_start + right.service

    missing = sort_items(required - visited)
    if missing:
        preview = ", ".join(stringify(customer_id) for customer_id in missing[:10])
        suffix = "..." if length(missing) > 10 else ""
        raise error(f"solution misses customers: {preview}{suffix}")

    vehicle_count = length(solution.routes)
    if vehicle_count == 0:
        raise error("solution contains no routes")
    return measurement_type(
        vehicle_count=vehicle_count,
        total_distance=round_distance(total_distance),
        mean_route_customers=instance.customer_count / vehicle_count,
    )


def _reject_metric_names(module: ModuleType) -> None:
    claimed = _SORTED(_PROTECTED_REPORT_NAMES.intersection(_VARS(module)))
    if claimed:
        raise EvalError(f"candidate declared self-reported metric names: {', '.join(claimed)}")


def _stop_payload(stop: Stop) -> dict[str, float | int]:
    return {
        "id": stop.customer_id,
        "x": stop.x,
        "y": stop.y,
        "demand": stop.demand,
        "earliest": stop.earliest,
        "latest": stop.latest,
        "service": stop.service,
    }


def _candidate_payload(instance: Instance) -> dict[str, object]:
    return {
        "cell": CELL.key,
        "name": instance.name,
        "seed": CELL.seed,
        "vehicle_limit": instance.vehicle_limit,
        "capacity": instance.capacity,
        "depot": _stop_payload(instance.depot),
        "customers": _TUPLE(_stop_payload(stop) for stop in instance.customers),
        "distance_convention": DISTANCE_CONVENTION,
        "objective": LEXICOGRAPHIC_OBJECTIVE,
    }


def _call_solver(
    solver: Callable[..., object],
    payload: dict[str, object],
    deadline: float,
    seed: int,
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
        raise EvalError("solve() must accept the instance as a positional argument")

    args: list[object] = [payload]
    kwargs: dict[str, object] = {}
    accepts_var_positional = _ANY(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    accepts_var_keyword = _ANY(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    for name, value in (("deadline", deadline), ("seed", seed)):
        parameter = _NEXT(
            (parameter for parameter in parameters if parameter.name == name),
            None,
        )
        if parameter is not None:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[name] = value
        elif accepts_var_positional:
            args.append(value)
        elif accepts_var_keyword:
            kwargs[name] = value
    try:
        return solver(*args, **kwargs)
    except _EXCEPTION as exc:
        raise EvalError(f"solve() raised: {exc}") from exc


def _load_candidate_solution(candidate_dir: Path, deadline: float) -> Solution:
    reject_metric_names = _reject_metric_names
    call_solver = _call_solver
    normalize_solution = _normalize_solution
    payload = _candidate_payload(INSTANCE)
    seed = CELL.seed
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise EvalError("candidate is missing solver.py")
    module_name = f"_autoevolve_vrp_{abs(hash(path.resolve()))}"
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
        reject_metric_names(module)
        solver = _GETATTR(module, "solve", None)
        if not _CALLABLE(solver):
            raise EvalError("solver.py must define callable solve()")
        raw = call_solver(solver, payload, deadline, seed)
        reject_metric_names(module)
        solution = normalize_solution(raw)
        reject_metric_names(module)
        return solution
    finally:
        sys.modules.pop(module_name, None)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run one deadline-aware candidate, recompute feasibility, and score it."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    candidate_budget = STAGES[stage].timeout_s - _DEADLINE_HEADROOM_S
    if candidate_budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")
    deadline = _MONOTONIC() + candidate_budget
    load_candidate_solution = _load_candidate_solution
    verify_solution = _verify_solution
    solution = load_candidate_solution(candidate_dir, deadline)
    measured = verify_solution(solution)
    return {
        GATE: 1.0,
        METRIC: measured.total_distance,
        "vehicle_count": _FLOAT(measured.vehicle_count),
        "mean_route_customers": measured.mean_route_customers,
        "customer_count": _FLOAT(INSTANCE.customer_count),
        "stage_reached": _FLOAT(stage),
    }


def ceiling() -> dict[str, float | str] | None:
    """No useful evaluator-computed distance lower bound is declared."""

    return None
