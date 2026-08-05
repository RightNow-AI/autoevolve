"""Exact exhaustive evaluator for fixed sorting networks."""

from __future__ import annotations

import importlib.util
import inspect
import operator
import os
import sys
import time
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="build-and-exhaustive-verify", timeout_s=120.0),
]
GATE = "sorts_all_binary_inputs"
METRIC = "size"
MAXIMIZE = False

DESCRIPTORS = [
    {"name": "depth", "metric": "depth", "bins": 12, "lo": 0.0, "hi": 36.0},
    {
        "name": "first_layer_channels",
        "metric": "first_layer_channels",
        "bins": 10,
        "lo": 0.0,
        "hi": 20.0,
    },
]

_CELLS = {
    "n11-validation": 11,
    "n13-frontier": 13,
    "n16-frontier": 16,
    "n20-frontier": 20,
}
_CELL = os.environ.get("AUTOEVOLVE_CELL", "n11-validation")
if _CELL not in _CELLS:
    _choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {_choices}; got {_CELL!r}")
CHANNELS = _CELLS[_CELL]

_SEARCH_BUDGET_S = STAGES[0].timeout_s * 0.75
# Candidate import happens before normalization and verification. Bind trusted
# primitives now so candidate code cannot replace a later gate dependency.
_BOOL = bool
_BYTES = bytes
_CALLABLE = callable
_ENUMERATE = enumerate
_EXCEPTION = Exception
_FLOAT = float
_GETATTR = getattr
_INDEX = operator.index
_INT = int
_ITER = iter
_NEXT = next
_LEN = len
_LIST = list
_MAX = max
_RANGE = range
_SET = set
_STOP_ITERATION = StopIteration
_TUPLE = tuple
_TYPE = type
_TYPE_ERROR = TypeError
_VALUE_ERROR = ValueError
_SIGNATURE = inspect.signature
_MONOTONIC = time.monotonic


def _make_binary_columns(channels: int) -> tuple[tuple[int, ...], int, int]:
    """Return one exact bit column per channel for all binary assignments."""

    input_count = 1 << channels
    byte_count = input_count // 8
    columns: list[int] = []
    byte_patterns = (0xAA, 0xCC, 0xF0)
    for channel in _RANGE(channels):
        if channel < 3:
            payload = _BYTES((byte_patterns[channel],)) * byte_count
        else:
            half_period = 1 << (channel - 3)
            period = b"\x00" * half_period + b"\xff" * half_period
            payload = period * (byte_count // _LEN(period))
        columns.append(_INT.from_bytes(payload, "little"))
    all_input_bits = (1 << input_count) - 1
    return tuple(columns), all_input_bits, input_count


_INPUT_COLUMNS, _ALL_INPUT_BITS, _BINARY_INPUT_COUNT = _make_binary_columns(CHANNELS)


def _snapshot_iterable(raw: object, field: str) -> tuple[object, ...]:
    """Consume candidate-controlled iteration once and retain plain immutable items."""

    try:
        iterator = _ITER(raw)
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be iterable, got {_TYPE(raw).__name__}") from exc
    items: list[object] = []
    index = 0
    while True:
        try:
            item = _NEXT(iterator)
        except _STOP_ITERATION:
            break
        except _EXCEPTION as exc:
            raise EvalError(f"{field} failed while reading item {index}: {exc}") from exc
        items.append(item)
        index += 1
    return _TUPLE(items)


def _exact_index(raw: object, field: str) -> int:
    if _TYPE(raw) is _BOOL:
        raise EvalError(f"{field} must be an integer, got bool")
    try:
        return _INT(_INDEX(raw))
    except _TYPE_ERROR as exc:
        raise EvalError(f"{field} must be an integer, got {_TYPE(raw).__name__}") from exc


def _normalize_network(raw: object) -> tuple[tuple[int, int], ...]:
    """Read the returned network once into plain integer comparator pairs."""

    raw_comparators = _snapshot_iterable(raw, "build() result")
    comparators: list[tuple[int, int]] = []
    for comparator_index, raw_comparator in _ENUMERATE(raw_comparators):
        pair = _snapshot_iterable(raw_comparator, f"comparator {comparator_index}")
        if _LEN(pair) != 2:
            raise EvalError(
                f"comparator {comparator_index} must contain exactly two indices"
            )
        left = _exact_index(pair[0], f"comparator {comparator_index} index 0")
        right = _exact_index(pair[1], f"comparator {comparator_index} index 1")
        for channel in (left, right):
            if not 0 <= channel < CHANNELS:
                raise EvalError(
                    f"comparator {comparator_index} channel {channel} is outside "
                    f"0..{CHANNELS - 1}"
                )
        if left == right:
            raise EvalError(
                f"comparator {comparator_index} compares channel {left} with itself"
            )
        if right < left:
            left, right = right, left
        comparators.append((left, right))
    return _TUPLE(comparators)


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "network.py"
    if not path.is_file():
        raise EvalError("candidate is missing network.py")
    spec = importlib.util.spec_from_file_location("sortnet_candidate", path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load network.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except _EXCEPTION as exc:
        raise EvalError(f"network.py failed to import: {exc}") from exc
    return module


def _call_build(module: ModuleType) -> tuple[tuple[int, int], ...]:
    build = _GETATTR(module, "build", None)
    if not _CALLABLE(build):
        raise EvalError("network.py must define callable build()")
    deadline = _MONOTONIC() + _SEARCH_BUDGET_S
    try:
        parameters = _SIGNATURE(build).parameters
    except (_TYPE_ERROR, _VALUE_ERROR):
        parameters = {}
    try:
        if _LEN(parameters) >= 2:
            raw = build(CHANNELS, deadline)
        else:
            raw = build(CHANNELS)
    except _EXCEPTION as exc:
        raise EvalError(f"build() raised: {exc}") from exc
    return _normalize_network(raw)


def _network_shape(network: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Return earliest legal depth and channels touched in its first layer."""

    last_layer = [-1] * CHANNELS
    first_layer_channels: set[int] = _SET()
    depth = 0
    for left, right in network:
        layer = _MAX(last_layer[left], last_layer[right]) + 1
        last_layer[left] = layer
        last_layer[right] = layer
        depth = _MAX(depth, layer + 1)
        if layer == 0:
            first_layer_channels.add(left)
            first_layer_channels.add(right)
    return depth, _LEN(first_layer_channels)


def _verify_all_binary_inputs(network: tuple[tuple[int, int], ...]) -> None:
    """Apply the network to all binary inputs in parallel and reject any inversion."""

    wires = _LIST(_INPUT_COLUMNS)
    for left, right in network:
        lower = wires[left] & wires[right]
        upper = wires[left] | wires[right]
        wires[left] = lower
        wires[right] = upper

    for channel in _RANGE(CHANNELS - 1):
        bad_inputs = wires[channel] & (_ALL_INPUT_BITS ^ wires[channel + 1])
        if bad_inputs:
            first_bad = (bad_inputs & -bad_inputs).bit_length() - 1
            binary = f"{first_bad:0{CHANNELS}b}"
            raise EvalError(
                f"network fails on binary input {binary}: output channels "
                f"{channel} and {channel + 1} contain 1,0"
            )


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Build once, normalize once, verify exhaustively, and score comparator count."""

    if stage < 0 or stage >= _LEN(STAGES):
        raise EvalError(f"unknown stage {stage}")
    module = _load_candidate(candidate_dir)
    network = _call_build(module)
    _verify_all_binary_inputs(network)
    depth, first_layer_channels = _network_shape(network)
    return {
        GATE: 1.0,
        METRIC: _FLOAT(_LEN(network)),
        "channels": _FLOAT(CHANNELS),
        "binary_inputs": _FLOAT(_BINARY_INPUT_COUNT),
        "depth": _FLOAT(depth),
        "first_layer_channels": _FLOAT(first_layer_channels),
    }


def ceiling() -> dict[str, float | str] | None:
    """No evaluator-computed lower ceiling exists for network size."""

    return None
