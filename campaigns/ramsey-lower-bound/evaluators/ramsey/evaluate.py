"""Exact lower-bound evaluator for diagonal Ramsey numbers."""

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

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="produce-and-gate", timeout_s=75.0),
    StageSpec(name="replay-and-total-recheck", timeout_s=180.0),
]
GATE = "mono_clique_free"
METRIC = "n_vertices"
MAXIMIZE = True

DESCRIPTORS = [
    {"name": "red_density", "metric": "red_density", "bins": 10, "lo": 0.0, "hi": 1.0},
    {
        "name": "certificate_family",
        "metric": "is_circulant",
        "bins": 2,
        "lo": -0.01,
        "hi": 1.01,
    },
]

_CELLS = {
    "k3-smoke": (3, 8, 5, 5),
    "k4-climb": (4, 24, 17, 17),
    "k5-frontier": (5, 50, 45, 41),
}
_CELL = os.environ.get("AUTOEVOLVE_CELL")
if _CELL not in _CELLS:
    choices = ", ".join(_CELLS)
    raise EvalError(f"AUTOEVOLVE_CELL must be one of {choices}; got {_CELL!r}")
S, CAP, CEILING_N, PERSIST_MIN_N = _CELLS[_CELL]

_PRODUCER_TIMEOUT_S = 45.0
_SEARCH_BUDGET_S = _PRODUCER_TIMEOUT_S * 0.75
_MAX_CERTIFICATE_BYTES = 1_000_000
_BITSET_OPERATION_LIMIT = 5_000_000
_EVALUATOR_DIR = Path(__file__).resolve().parent
_PRODUCER = _EVALUATOR_DIR / "produce.py"
_CERTIFICATE_ROOT = _EVALUATOR_DIR / "certificates"
_CHILD_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "PYTHONIOENCODING",
)
_CHILD_ENV = {name: os.environ[name] for name in _CHILD_ENV_KEYS if name in os.environ}
_CHILD_ENV["PYTHONHASHSEED"] = "0"
_CHILD_ENV["PYTHONIOENCODING"] = "utf-8"
_CHILD_ENV["PYTHONSAFEPATH"] = "1"


@dataclass(frozen=True)
class Certificate:
    form: str
    n: int
    red_diffs: tuple[int, ...] = ()
    red_edges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Verdict:
    valid: bool
    color: str | None = None
    vertices: tuple[int, ...] = ()


class _GateBudgetExhausted(RuntimeError):
    pass


@dataclass
class _OperationBudget:
    remaining: int

    def spend(self) -> None:
        if self.remaining <= 0:
            raise _GateBudgetExhausted
        self.remaining -= 1


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
    form = values.get("form")
    if form == "circulant":
        expected = {"form", "n", "red_diffs"}
    elif form == "adjacency":
        expected = {"form", "n", "red_edges"}
    else:
        raise EvalError("certificate form must be exactly 'circulant' or 'adjacency'")
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
    if not S <= n <= CAP:
        raise EvalError(f"n must satisfy {S} <= n <= {CAP}, got {n}")

    if form == "circulant":
        raw_diffs = _sequence_snapshot(values["red_diffs"], "red_diffs")
        differences = tuple(
            _exact_int(value, f"red_diffs[{index}]")
            for index, value in enumerate(raw_diffs)
        )
        previous = 0
        for index, difference in enumerate(differences):
            if not 1 <= difference <= n // 2:
                raise EvalError(
                    f"red_diffs[{index}] must satisfy 1 <= d <= {n // 2}, got {difference}"
                )
            if difference <= previous:
                raise EvalError("red_diffs must be strictly increasing")
            previous = difference
        return Certificate(form="circulant", n=n, red_diffs=differences)

    raw_edges = _sequence_snapshot(values["red_edges"], "red_edges")
    if len(raw_edges) > n * (n - 1) // 2:
        raise EvalError("red_edges is longer than the complete graph edge set")
    edges: list[tuple[int, int]] = []
    previous_edge: tuple[int, int] | None = None
    for index, raw_edge in enumerate(raw_edges):
        pair = _sequence_snapshot(raw_edge, f"red_edges[{index}]")
        if len(pair) != 2:
            raise EvalError(f"red_edges[{index}] must contain exactly two integers")
        left = _exact_int(pair[0], f"red_edges[{index}][0]")
        right = _exact_int(pair[1], f"red_edges[{index}][1]")
        edge = (left, right)
        if not 0 <= left < right < n:
            raise EvalError(f"red_edges[{index}] must satisfy 0 <= i < j < {n}")
        if previous_edge is not None and edge <= previous_edge:
            raise EvalError("red_edges must be strictly increasing in lexicographic order")
        edges.append(edge)
        previous_edge = edge
    return Certificate(form="adjacency", n=n, red_edges=tuple(edges))


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


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


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
        str(S),
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


def _expand_masks(certificate: Certificate) -> tuple[tuple[int, ...], tuple[int, ...]]:
    red = [0] * certificate.n
    if certificate.form == "circulant":
        red_diffs = frozenset(certificate.red_diffs)
        for left in range(certificate.n):
            for right in range(left + 1, certificate.n):
                raw_difference = right - left
                difference = min(raw_difference, certificate.n - raw_difference)
                if difference in red_diffs:
                    red[left] |= 1 << right
                    red[right] |= 1 << left
    else:
        for left, right in certificate.red_edges:
            red[left] |= 1 << right
            red[right] |= 1 << left

    all_vertices = (1 << certificate.n) - 1
    blue = [all_vertices & ~(1 << vertex) & ~red[vertex] for vertex in range(certificate.n)]
    for vertex in range(certificate.n):
        allowed = all_vertices ^ (1 << vertex)
        if red[vertex] & (1 << vertex) or blue[vertex] & (1 << vertex):
            raise EvalError("decoded coloring contains a self loop")
        if red[vertex] & blue[vertex]:
            raise EvalError("decoded coloring assigns one pair both colors")
        if (red[vertex] | blue[vertex]) != allowed:
            raise EvalError("decoded coloring does not assign every pair exactly one color")
        for other in range(certificate.n):
            if bool(red[vertex] & (1 << other)) != bool(red[other] & (1 << vertex)):
                raise EvalError("decoded red adjacency is not symmetric")
            if bool(blue[vertex] & (1 << other)) != bool(blue[other] & (1 << vertex)):
                raise EvalError("decoded blue adjacency is not symmetric")
    return tuple(red), tuple(blue)


def _find_clique(
    neighbors: tuple[int, ...],
    clique_size: int,
    operation_limit: int,
) -> tuple[int, ...] | None:
    budget = _OperationBudget(operation_limit)

    def search(candidates: int, needed: int, chosen: tuple[int, ...]) -> tuple[int, ...] | None:
        budget.spend()
        if needed == 0:
            return chosen
        while candidates.bit_count() >= needed:
            vertex_bit = candidates & -candidates
            candidates ^= vertex_bit
            vertex = vertex_bit.bit_length() - 1
            found = search(candidates & neighbors[vertex], needed - 1, (*chosen, vertex))
            if found is not None:
                return found
        return None

    return search((1 << len(neighbors)) - 1, clique_size, ())


def _bitset_verdict(
    red: tuple[int, ...],
    blue: tuple[int, ...],
    clique_size: int,
    operation_limit: int = _BITSET_OPERATION_LIMIT,
) -> Verdict:
    """Return the exact bitset verdict, or fail closed if its budget is exhausted."""

    try:
        red_witness = _find_clique(red, clique_size, operation_limit)
        blue_witness = _find_clique(blue, clique_size, operation_limit)
    except _GateBudgetExhausted as exc:
        raise EvalError("bitset clique gate operation budget exhausted; rejected closed") from exc
    if red_witness is not None:
        return Verdict(valid=False, color="RED", vertices=red_witness)
    if blue_witness is not None:
        return Verdict(valid=False, color="BLUE", vertices=blue_witness)
    return Verdict(valid=True)


def _is_red_from_certificate(
    certificate: Certificate,
    left: int,
    right: int,
    red_diff_set: frozenset[int],
    red_edge_set: frozenset[tuple[int, int]],
) -> bool:
    if certificate.form == "circulant":
        raw_difference = (left - right) % certificate.n
        difference = min(raw_difference, (-raw_difference) % certificate.n)
        return difference in red_diff_set
    edge = (left, right) if left < right else (right, left)
    return edge in red_edge_set


def _naive_verdict(certificate: Certificate) -> Verdict:
    """Independently enumerate every target subset from the immutable certificate."""

    red_diff_set = frozenset(certificate.red_diffs)
    red_edge_set = frozenset(certificate.red_edges)
    for vertices in itertools.combinations(range(certificate.n), S):
        all_red = True
        all_blue = True
        for left, right in itertools.combinations(vertices, 2):
            is_red = _is_red_from_certificate(
                certificate,
                left,
                right,
                red_diff_set,
                red_edge_set,
            )
            all_red = all_red and is_red
            all_blue = all_blue and not is_red
            if not all_red and not all_blue:
                break
        if all_red:
            return Verdict(valid=False, color="RED", vertices=vertices)
        if all_blue:
            return Verdict(valid=False, color="BLUE", vertices=vertices)
    return Verdict(valid=True)


def _raise_invalid(verdict: Verdict) -> None:
    vertices = ",".join(str(vertex) for vertex in verdict.vertices)
    raise EvalError(f"{verdict.color} contains K{S} on vertices [{vertices}]")


def _canonical_bytes(certificate: Certificate) -> bytes:
    if certificate.form == "circulant":
        payload: dict[str, object] = {
            "form": "circulant",
            "n": certificate.n,
            "red_diffs": list(certificate.red_diffs),
        }
    else:
        payload = {
            "form": "adjacency",
            "n": certificate.n,
            "red_edges": [list(edge) for edge in certificate.red_edges],
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
    red: tuple[int, ...],
    stage: int,
    persisted: bool,
) -> dict[str, float]:
    red_edge_count = sum(mask.bit_count() for mask in red) // 2
    total_edges = certificate.n * (certificate.n - 1) // 2
    return {
        GATE: 1.0,
        METRIC: float(certificate.n),
        "red_edge_count": float(red_edge_count),
        "red_density": red_edge_count / total_edges,
        "is_circulant": 1.0 if certificate.form == "circulant" else 0.0,
        "diff_class_count": (
            float(len(certificate.red_diffs)) if certificate.form == "circulant" else -1.0
        ),
        "target_clique": float(S),
        "certificate_persisted": 1.0 if persisted else 0.0,
        "replay_identical": 1.0 if stage == 1 else 0.0,
        "stage_reached": float(stage),
    }


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Produce, normalize, and exactly verify one Ramsey coloring certificate."""

    if stage not in (0, 1):
        raise EvalError(f"unknown stage {stage}")
    with tempfile.TemporaryDirectory(prefix="autoevolve-ramsey-") as temporary_raw:
        temporary = Path(temporary_raw)
        first = _produce(candidate_dir, temporary / "first.json", temporary / "first.err")
        first_red, first_blue = _expand_masks(first)
        first_fast = _bitset_verdict(first_red, first_blue, S)
        if not first_fast.valid:
            _raise_invalid(first_fast)
        if stage == 0:
            return _metrics(first, first_red, stage, persisted=False)

        second = _produce(candidate_dir, temporary / "second.json", temporary / "second.err")
        if first != second:
            raise EvalError("construct() replay differed across two fresh interpreters")
        second_red, second_blue = _expand_masks(second)
        second_fast = _bitset_verdict(second_red, second_blue, S)
        slow = _naive_verdict(first)
        if first_fast.valid != slow.valid or second_fast.valid != slow.valid:
            raise EvalError("independent Ramsey verifiers disagreed; rejected closed")
        if not second_fast.valid:
            _raise_invalid(second_fast)
        if not slow.valid:
            _raise_invalid(slow)
        persisted_path = _persist_certificate(first)
        return _metrics(first, first_red, stage, persisted=persisted_path is not None)


def ceiling() -> dict[str, float | str]:
    """Return the cell's cited maximum possible certificate size."""

    methods = {
        "k3-smoke": "R(3,3)=6, so a valid certificate has at most 5 vertices",
        "k4-climb": "Greenwood and Gleason proved R(4,4)=18",
        "k5-frontier": "Angeltveit and McKay proved R(5,5)<=46",
    }
    return {"metric": METRIC, "value": float(CEILING_N), "method": methods[_CELL]}
