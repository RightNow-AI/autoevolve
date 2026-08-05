"""Exact evaluator for multicolor Ramsey lower-bound certificates."""

from __future__ import annotations

import hashlib
import itertools
import json
import operator
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from autoevolve.eval.childenv import build_child_env
from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="produce-and-gate", timeout_s=75.0),
    StageSpec(name="replay-and-exhaustive-recheck", timeout_s=180.0),
]
GATE = "forbidden_subgraph_free"
METRIC = "n_vertices"
MAXIMIZE = True

DESCRIPTORS = [
    {"name": "red_density", "metric": "red_density", "bins": 10, "lo": 0.0, "hi": 1.0},
    {
        "name": "distinct_color_class_sizes",
        "metric": "distinct_color_class_sizes",
        "bins": 4,
        "lo": 0.99,
        "hi": 4.01,
    },
]

_CELLS = {
    "n5-validation": (5, 5),
    "n49-frontier": (49, 48),
}
_CELL = os.environ.get("AUTOEVOLVE_CELL")
if _CELL not in _CELLS:
    choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {choices}; got {_CELL!r}")
CAP, PERSIST_MIN_N = _CELLS[_CELL]

MIN_N = 5
RED = 0
BLUE = 1
GREEN = 2
YELLOW = 3
COLOR_NAMES = ("RED", "BLUE", "GREEN", "YELLOW")

_PRODUCER_TIMEOUT_S = 45.0
_SEARCH_BUDGET_S = _PRODUCER_TIMEOUT_S * 0.75
_MAX_CERTIFICATE_BYTES = 1_000_000
_EVALUATOR_DIR = Path(__file__).resolve().parent
_PRODUCER = _EVALUATOR_DIR / "produce.py"
_CERTIFICATE_ROOT = _EVALUATOR_DIR / "certificates"
_CHILD_ENV = build_child_env()


@dataclass(frozen=True)
class Certificate:
    n: int
    edge_colors: tuple[int, ...]


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str | None = None
    vertices: tuple[int, ...] = ()


def _mapping_snapshot(raw: object, field: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise EvalError(f"{field} must be an object, got {type(raw).__name__}")
    try:
        items = tuple(raw.items())
    except Exception as exc:
        raise EvalError(f"{field} could not be read once: {exc}") from exc
    snapshot: dict[str, object] = {}
    for index, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise EvalError(f"{field}.items()[{index}] is not a key-value pair")
        key, value = item
        if type(key) is not str:
            raise EvalError(f"{field} key {index} must be a plain string")
        if key in snapshot:
            raise EvalError(f"{field} contains duplicate key {key!r}")
        snapshot[key] = value
    return snapshot


def _sequence_snapshot(raw: object, field: str) -> tuple[object, ...]:
    if isinstance(raw, str | bytes | bytearray) or not isinstance(raw, list | tuple):
        raise EvalError(f"{field} must be an array")
    try:
        return tuple(raw)
    except Exception as exc:
        raise EvalError(f"{field} could not be read once: {exc}") from exc


def _exact_int(raw: object, field: str) -> int:
    if isinstance(raw, bool):
        raise EvalError(f"{field} must be an integer, got bool")
    try:
        return int(operator.index(raw))
    except TypeError as exc:
        raise EvalError(f"{field} must be an integer, got {type(raw).__name__}") from exc


def _normalize_certificate(raw: object) -> Certificate:
    """Read a certificate once and retain only immutable primitive values."""

    values = _mapping_snapshot(raw, "certificate")
    expected = {"n", "edge_colors"}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise EvalError(f"certificate schema is exact; {'; '.join(details)}")

    n = _exact_int(values["n"], "n")
    if not MIN_N <= n <= CAP:
        raise EvalError(f"n must satisfy {MIN_N} <= n <= {CAP}, got {n}")

    raw_colors = _sequence_snapshot(values["edge_colors"], "edge_colors")
    expected_edges = n * (n - 1) // 2
    if len(raw_colors) != expected_edges:
        raise EvalError(
            f"edge_colors must contain exactly {expected_edges} entries, got {len(raw_colors)}"
        )
    colors = tuple(
        _exact_int(value, f"edge_colors[{index}]")
        for index, value in enumerate(raw_colors)
    )
    for index, color in enumerate(colors):
        if not RED <= color <= YELLOW:
            raise EvalError(f"edge_colors[{index}] must be in 0..3, got {color}")
    return Certificate(n=n, edge_colors=colors)


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _read_certificate(path: Path) -> Certificate:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise EvalError(f"certificate file is absent or unreadable: {exc}") from exc
    if size > _MAX_CERTIFICATE_BYTES:
        raise EvalError(
            f"certificate exceeds {_MAX_CERTIFICATE_BYTES} bytes: reported size {size}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"certificate file is unreadable: {exc}") from exc
    if len(payload) > _MAX_CERTIFICATE_BYTES:
        raise EvalError(f"certificate exceeds {_MAX_CERTIFICATE_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvalError("certificate is not valid UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise EvalError(f"certificate is not valid JSON: {exc}") from exc
    return _normalize_certificate(raw)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            process.kill()


def _produce(candidate_dir: Path, output_path: Path, stderr_path: Path) -> Certificate:
    command = [
        sys.executable,
        "-P",
        "-s",
        "-B",
        str(_PRODUCER),
        str(candidate_dir),
        str(output_path),
        str(CAP),
        str(_SEARCH_BUDGET_S),
    ]
    with stderr_path.open("wb") as error_handle:
        common = {
            "cwd": candidate_dir,
            "env": _CHILD_ENV,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": error_handle,
        }
        try:
            if sys.platform == "win32":
                process = subprocess.Popen(
                    command,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    **common,
                )
            else:
                process = subprocess.Popen(command, start_new_session=True, **common)
        except OSError as exc:
            raise EvalError(f"could not start construct() producer: {exc}") from exc
        try:
            return_code = process.wait(timeout=_PRODUCER_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    process.kill()
            raise EvalError(f"construct() exceeded {_PRODUCER_TIMEOUT_S:g}s") from exc
    if return_code != 0:
        try:
            tail = stderr_path.read_bytes()[-2000:].decode("utf-8", errors="replace").strip()
        except OSError:
            tail = "unreadable stderr"
        raise EvalError(f"construct() producer exited with code {return_code}: {tail}")
    return _read_certificate(output_path)


def _edge_index(n: int, left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    return left * (2 * n - left - 1) // 2 + right - left - 1


def _color(certificate: Certificate, left: int, right: int) -> int:
    return certificate.edge_colors[_edge_index(certificate.n, left, right)]


def _expand_masks(certificate: Certificate) -> tuple[tuple[int, ...], ...]:
    masks = [[0] * certificate.n for _ in range(4)]
    for left, right in itertools.combinations(range(certificate.n), 2):
        color = _color(certificate, left, right)
        masks[color][left] |= 1 << right
        masks[color][right] |= 1 << left

    all_vertices = (1 << certificate.n) - 1
    for vertex in range(certificate.n):
        allowed = all_vertices ^ (1 << vertex)
        union = 0
        for color in range(4):
            row = masks[color][vertex]
            if row & (1 << vertex):
                raise EvalError("decoded coloring contains a self loop")
            if union & row:
                raise EvalError("decoded coloring assigns one pair multiple colors")
            union |= row
            for other in range(certificate.n):
                if bool(row & (1 << other)) != bool(masks[color][other] & (1 << vertex)):
                    raise EvalError("decoded color adjacency is not symmetric")
        if union != allowed:
            raise EvalError("decoded coloring does not assign every pair exactly one color")
    return tuple(tuple(rows) for rows in masks)


def _first_two_vertices(mask: int) -> tuple[int, int]:
    first_bit = mask & -mask
    second_bit = (mask ^ first_bit) & -(mask ^ first_bit)
    return first_bit.bit_length() - 1, second_bit.bit_length() - 1


def _fast_verdict(
    certificate: Certificate,
    masks: tuple[tuple[int, ...], ...],
) -> Verdict:
    """Check all four forbidden structures with exact bitset identities."""

    red = masks[RED]
    for left in range(certificate.n):
        neighbors = red[left] & ~((1 << (left + 1)) - 1)
        while neighbors:
            right_bit = neighbors & -neighbors
            neighbors ^= right_bit
            right = right_bit.bit_length() - 1
            common = red[left] & red[right]
            if common:
                third = (common & -common).bit_length() - 1
                return Verdict(False, "RED contains K3", (left, right, third))

    blue = masks[BLUE]
    for first in range(certificate.n):
        second_bits = blue[first] & ~((1 << (first + 1)) - 1)
        while second_bits:
            second_bit = second_bits & -second_bits
            second_bits ^= second_bit
            second = second_bit.bit_length() - 1
            common_pair = blue[first] & blue[second] & ~((1 << (second + 1)) - 1)
            third_bits = common_pair
            while third_bits:
                third_bit = third_bits & -third_bits
                third_bits ^= third_bit
                third = third_bit.bit_length() - 1
                fourth_bits = common_pair & blue[third] & ~((1 << (third + 1)) - 1)
                if fourth_bits:
                    fourth = (fourth_bits & -fourth_bits).bit_length() - 1
                    return Verdict(False, "BLUE contains K4", (first, second, third, fourth))

    for color in (GREEN, YELLOW):
        adjacency = masks[color]
        for left, right in itertools.combinations(range(certificate.n), 2):
            common = adjacency[left] & adjacency[right]
            if common.bit_count() >= 2:
                first, second = _first_two_vertices(common)
                reason = f"{COLOR_NAMES[color]} contains C4"
                return Verdict(False, reason, (left, first, right, second))
    return Verdict(True)


def _cycle_is_color(
    certificate: Certificate,
    cycle: tuple[int, int, int, int],
    color: int,
) -> bool:
    return all(
        _color(certificate, cycle[index], cycle[(index + 1) % 4]) == color
        for index in range(4)
    )


def _exhaustive_verdict(certificate: Certificate) -> Verdict:
    """Independently enumerate every forbidden subgraph from edge colors."""

    for vertices in itertools.combinations(range(certificate.n), 3):
        if all(
            _color(certificate, left, right) == RED
            for left, right in itertools.combinations(vertices, 2)
        ):
            return Verdict(False, "RED contains K3", vertices)

    for vertices in itertools.combinations(range(certificate.n), 4):
        if all(
            _color(certificate, left, right) == BLUE
            for left, right in itertools.combinations(vertices, 2)
        ):
            return Verdict(False, "BLUE contains K4", vertices)
        first, second, third, fourth = vertices
        cycles = (
            (first, second, third, fourth),
            (first, second, fourth, third),
            (first, third, second, fourth),
        )
        for color in (GREEN, YELLOW):
            for cycle in cycles:
                if _cycle_is_color(certificate, cycle, color):
                    reason = f"{COLOR_NAMES[color]} contains C4"
                    return Verdict(False, reason, cycle)
    return Verdict(True)


def _raise_invalid(verdict: Verdict) -> None:
    vertices = ",".join(str(vertex) for vertex in verdict.vertices)
    raise EvalError(f"{verdict.reason} on vertices [{vertices}]")


def _canonical_bytes(certificate: Certificate) -> bytes:
    payload = {
        "n": certificate.n,
        "edge_colors": list(certificate.edge_colors),
    }
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )
    return text.encode("utf-8")


def _persist_certificate(certificate: Certificate) -> Path | None:
    if certificate.n < PERSIST_MIN_N:
        return None
    payload = _canonical_bytes(certificate)
    digest = hashlib.sha256(payload).hexdigest()
    directory = _CERTIFICATE_ROOT / _CELL
    path = directory / f"n{certificate.n}-{digest[:16]}.json"
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if path.read_bytes() != payload:
                raise EvalError(f"certificate artifact collision at {path}")
            return path
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except EvalError:
        raise
    except OSError as exc:
        raise EvalError(f"could not persist canonical certificate: {exc}") from exc
    return path


def _metrics(
    certificate: Certificate,
    stage: int,
    persisted: bool,
) -> dict[str, float]:
    color_sizes = [certificate.edge_colors.count(color) for color in range(4)]
    total_edges = len(certificate.edge_colors)
    return {
        GATE: 1.0,
        METRIC: float(certificate.n),
        "red_edges": float(color_sizes[RED]),
        "blue_edges": float(color_sizes[BLUE]),
        "green_edges": float(color_sizes[GREEN]),
        "yellow_edges": float(color_sizes[YELLOW]),
        "red_density": color_sizes[RED] / total_edges,
        "distinct_color_class_sizes": float(len(set(color_sizes))),
        "certificate_persisted": 1.0 if persisted else 0.0,
        "replay_identical": 1.0 if stage == 1 else 0.0,
        "stage_reached": float(stage),
    }


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Produce, normalize, and exactly verify one four-color certificate."""

    if stage not in (0, 1):
        raise EvalError(f"unknown stage {stage}")
    with tempfile.TemporaryDirectory(prefix="autoevolve-multiramsey-") as raw:
        temporary = Path(raw)
        first = _produce(candidate_dir, temporary / "first.json", temporary / "first.err")
        first_masks = _expand_masks(first)
        first_fast = _fast_verdict(first, first_masks)
        if not first_fast.valid:
            _raise_invalid(first_fast)
        if stage == 0:
            return _metrics(first, stage, persisted=False)

        second = _produce(candidate_dir, temporary / "second.json", temporary / "second.err")
        if first != second:
            raise EvalError("construct() replay differed across two fresh interpreters")
        second_masks = _expand_masks(second)
        second_fast = _fast_verdict(second, second_masks)
        exhaustive = _exhaustive_verdict(first)
        verdicts = (first_fast.valid, second_fast.valid, exhaustive.valid)
        if len(set(verdicts)) != 1:
            raise EvalError("independent multicolor Ramsey verifiers disagreed; rejected closed")
        if not second_fast.valid:
            _raise_invalid(second_fast)
        if not exhaustive.valid:
            _raise_invalid(exhaustive)
        persisted_path = _persist_certificate(first)
        return _metrics(first, stage, persisted=persisted_path is not None)


def ceiling() -> dict[str, float | str]:
    """Return the largest certificate size permitted by the cited upper bound."""

    return {
        "metric": METRIC,
        "value": 74.0,
        "method": "the published upper bound R(K3,K4,C4,C4) <= 75",
    }
