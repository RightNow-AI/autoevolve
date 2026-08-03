"""Exact evaluator for Frankl's union-closed sets conjecture."""

from __future__ import annotations

import importlib.util
import inspect
import json
import operator
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="exact-gate", timeout_s=90.0),
    StageSpec(name="replay-and-cross-check", timeout_s=240.0),
]
GATE = "union_closed_valid"
METRIC = "max_freq_ratio"
MAXIMIZE = False

_MODEL_MAX_BYTES = 1 << 20
_CERTIFICATE_MAX_BYTES = 4 << 20
_HARD_N_MAX = 24
_HARD_M_MAX = 20_000
_HALF = Fraction(1, 2)

_BITWISE_OR = np.bitwise_or
_FLATNONZERO = np.flatnonzero
_LOGICAL_NOT = np.logical_not
_NP_ARRAY = np.array
_NP_BOOL = np.bool_
_NP_BOOL_TYPE = type(np.bool_(False))
_NP_UINT32 = np.uint32
_NP_ZEROS = np.zeros
_JSON_DUMPS = json.dumps
_MODULE_FROM_SPEC = importlib.util.module_from_spec
_OPERATOR_INDEX = operator.index
_PARAMETER = inspect.Parameter
_SIGNATURE = inspect.signature
_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory
_MONOTONIC = time.monotonic
_PATH_WRITE_BYTES = Path.write_bytes


def _read_cap(name: str, hard_cap: int) -> tuple[int, str | None]:
    """Read one tightening-only workload cap before candidate code can run."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return hard_cap, None
    text = raw.strip()
    if not text.isdecimal():
        return hard_cap, f"{name} is not a base-10 integer"
    value = int(text, 10)
    if value < 1:
        return hard_cap, f"{name} must be >= 1, got {value}"
    return min(hard_cap, value), None


N_MAX, _N_MAX_ERROR = _read_cap("AUTOEVOLVE_UCF_NMAX", _HARD_N_MAX)
M_MAX, _M_MAX_ERROR = _read_cap("AUTOEVOLVE_UCF_MMAX", _HARD_M_MAX)
CONFIG_ERROR = _N_MAX_ERROR or _M_MAX_ERROR
_CLOSURE_PAIR_BUDGET = M_MAX * (M_MAX + 1) // 2


@dataclass(frozen=True)
class _Certificate:
    n: int
    sets: tuple[int, ...]
    wire: bytes


@dataclass(frozen=True)
class _Measurement:
    frequencies: tuple[int, ...]
    max_freq: int
    used_elements: int

    def metrics(self, certificate: _Certificate) -> dict[str, float]:
        """Return display metrics after all exact gate decisions are complete."""

        family_size = len(certificate.sets)
        ratio = Fraction(self.max_freq, family_size)
        below_half = ratio < _HALF
        return {
            GATE: 1.0,
            METRIC: float(ratio),
            "max_freq": float(self.max_freq),
            "family_size": float(family_size),
            "ground_set_declared": float(certificate.n),
            "ground_set_used": float(self.used_elements),
            "below_half": float(below_half),
            "half_margin": float(2 * self.max_freq - family_size),
            "nmax_in_force": float(N_MAX),
        }


def _exact_int(value: object, label: str) -> int:
    """Normalize one integer, accepting numpy integers but never booleans."""

    if isinstance(value, bool | _NP_BOOL_TYPE):
        raise EvalError(f"{label} must be an integer, got bool")
    try:
        return int(_OPERATOR_INDEX(value))
    except TypeError as exc:
        raise EvalError(f"{label} must be an integer, got {type(value).__name__}") from exc


def _wire_certificate(n: int, sets: tuple[int, ...]) -> bytes:
    payload = {"n": n, "sets": list(sets)}
    wire = _JSON_DUMPS(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(wire) > _CERTIFICATE_MAX_BYTES:
        raise EvalError(
            f"normalized certificate is {len(wire)} bytes; limit is {_CERTIFICATE_MAX_BYTES}"
        )
    return wire


def _normalize_certificate(raw: object) -> _Certificate:
    """Read a candidate mapping and its set sequence once into immutable primitives."""

    if not isinstance(raw, Mapping):
        raise EvalError(
            f"build_family() must return a mapping, got {type(raw).__name__}"
        )
    try:
        iterator = iter(raw.items())
        entries: list[object] = []
        for _ in range(3):
            try:
                entries.append(next(iterator))
            except StopIteration:
                break
    except Exception as exc:
        raise EvalError(f"build_family() returned an unreadable mapping: {exc}") from exc
    if len(entries) != 2:
        raise EvalError(f"certificate must contain exactly 2 keys, got {len(entries)}")

    values: dict[str, object] = {}
    for entry in entries:
        try:
            key, value = entry  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise EvalError("certificate mapping items must be key-value pairs") from exc
        if not isinstance(key, str):
            raise EvalError(f"certificate keys must be strings, got {type(key).__name__}")
        if key in values:
            raise EvalError(f"certificate key {key!r} appears more than once")
        values[key] = value
    if set(values) != {"n", "sets"}:
        found = ", ".join(sorted(values))
        raise EvalError(f"certificate keys must be exactly n and sets; got {found}")

    n = _exact_int(values["n"], "n")
    if not 1 <= n <= N_MAX:
        raise EvalError(f"n must satisfy 1 <= n <= {N_MAX}, got {n}")
    raw_sets = values["sets"]
    if isinstance(raw_sets, str | bytes | bytearray | Mapping):
        raise EvalError("sets must be an array-like iterable of integers")
    try:
        set_iterator = iter(raw_sets)  # type: ignore[arg-type]
    except TypeError as exc:
        raise EvalError("sets must be an array-like iterable of integers") from exc

    normalized: list[int] = []
    previous = -1
    for index in range(M_MAX + 1):
        try:
            item = next(set_iterator)
        except StopIteration:
            break
        if index == M_MAX:
            raise EvalError(f"sets may contain at most {M_MAX} members")
        mask = _exact_int(item, f"sets[{index}]")
        if not 0 <= mask < (1 << n):
            raise EvalError(f"sets[{index}]={mask} is outside [0, {1 << n})")
        if mask <= previous:
            raise EvalError(
                f"sets must be strictly ascending; sets[{index - 1}]={previous} "
                f">= sets[{index}]={mask}"
            )
        normalized.append(mask)
        previous = mask
    if not normalized:
        raise EvalError("sets must contain at least one member; the empty family is excluded")
    if normalized[-1] == 0:
        raise EvalError(
            "sets must contain at least one nonempty member; {empty set} is excluded"
        )
    sets = tuple(normalized)
    return _Certificate(n=n, sets=sets, wire=_wire_certificate(n, sets))


def _normalize_certificate_replay(raw: object) -> _Certificate:
    """Independently normalize the replay result with a separate read path."""

    if not isinstance(raw, Mapping):
        raise EvalError("replay build_family() must return a mapping")
    replay_items: list[tuple[object, object]] = []
    try:
        items = iter(raw.items())
        while len(replay_items) <= 2:
            try:
                item = next(items)
            except StopIteration:
                break
            try:
                key, value = item
            except (TypeError, ValueError) as exc:
                raise EvalError("replay mapping items must be key-value pairs") from exc
            replay_items.append((key, value))
    except EvalError:
        raise
    except Exception as exc:
        raise EvalError(f"replay mapping could not be read: {exc}") from exc
    if len(replay_items) != 2:
        raise EvalError("replay certificate must contain exactly the keys n and sets")

    replay_values: dict[str, object] = {}
    for key, value in replay_items:
        if not isinstance(key, str) or key not in {"n", "sets"} or key in replay_values:
            raise EvalError("replay certificate must contain exactly the keys n and sets")
        replay_values[key] = value
    if len(replay_values) != 2:
        raise EvalError("replay certificate must contain exactly the keys n and sets")

    n_value = replay_values["n"]
    if isinstance(n_value, bool | _NP_BOOL_TYPE):
        raise EvalError("replay n must be an integer, not bool")
    try:
        n = int(_OPERATOR_INDEX(n_value))
    except TypeError as exc:
        raise EvalError("replay n must be an integer") from exc
    if n < 1 or n > N_MAX:
        raise EvalError(f"replay n must satisfy 1 <= n <= {N_MAX}, got {n}")

    set_source = replay_values["sets"]
    if isinstance(set_source, str | bytes | bytearray | Mapping):
        raise EvalError("replay sets must be an array-like iterable")
    try:
        source_iterator = iter(set_source)  # type: ignore[arg-type]
    except TypeError as exc:
        raise EvalError("replay sets must be an array-like iterable") from exc
    replay_sets: list[int] = []
    last: int | None = None
    while len(replay_sets) <= M_MAX:
        try:
            value = next(source_iterator)
        except StopIteration:
            break
        if len(replay_sets) == M_MAX:
            raise EvalError(f"replay sets may contain at most {M_MAX} members")
        if isinstance(value, bool | _NP_BOOL_TYPE):
            raise EvalError(f"replay sets[{len(replay_sets)}] must not be bool")
        try:
            mask = int(_OPERATOR_INDEX(value))
        except TypeError as exc:
            raise EvalError(f"replay sets[{len(replay_sets)}] must be an integer") from exc
        if mask < 0 or mask >= 1 << n:
            raise EvalError(f"replay set mask {mask} is out of range")
        if last is not None and mask <= last:
            raise EvalError("replay sets must be strictly ascending")
        replay_sets.append(mask)
        last = mask
    if not replay_sets:
        raise EvalError("replay rejected the empty family")
    if replay_sets[-1] == 0:
        raise EvalError("replay rejected the family containing only the empty set")
    frozen_sets = tuple(replay_sets)
    return _Certificate(n=n, sets=frozen_sets, wire=_wire_certificate(n, frozen_sets))


def _call_builder(builder: Callable[..., object], deadline: float) -> object:
    """Call a zero-argument builder or pass the documented deadline when accepted."""

    try:
        parameters = list(_SIGNATURE(builder).parameters.values())
    except (TypeError, ValueError) as exc:
        raise EvalError(f"could not inspect build_family() signature: {exc}") from exc
    positional = any(
        parameter.kind
        in (
            _PARAMETER.POSITIONAL_ONLY,
            _PARAMETER.POSITIONAL_OR_KEYWORD,
            _PARAMETER.VAR_POSITIONAL,
        )
        for parameter in parameters
    )
    keyword_deadline = any(
        parameter.name == "deadline"
        and parameter.kind is _PARAMETER.KEYWORD_ONLY
        for parameter in parameters
    )
    arbitrary_keywords = any(
        parameter.kind is _PARAMETER.VAR_KEYWORD for parameter in parameters
    )
    if positional:
        return builder(deadline)
    if keyword_deadline or arbitrary_keywords:
        return builder(deadline=deadline)
    return builder()


def _extract_certificate(
    source: bytes,
    deadline: float,
    nonce: str,
    normalizer: Callable[[object], _Certificate],
) -> _Certificate:
    """Copy only model.py, execute it by explicit path, and snapshot its result."""

    with _TEMPORARY_DIRECTORY(prefix="autoevolve-ucf-") as temp_name:
        model_path = Path(temp_name) / "model.py"
        _PATH_WRITE_BYTES(model_path, source)
        module_name = f"_autoevolve_ucf_candidate_{nonce}"
        spec = _SPEC_FROM_FILE_LOCATION(module_name, model_path)
        if spec is None or spec.loader is None:
            raise EvalError("could not load candidate model.py")
        module: ModuleType = _MODULE_FROM_SPEC(spec)
        sys.modules[module_name] = module
        try:
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise EvalError(f"model.py failed to import: {exc}") from exc
            builder = getattr(module, "build_family", None)
            if not callable(builder):
                raise EvalError("model.py must define callable build_family()")
            try:
                raw = _call_builder(builder, deadline)
            except EvalError:
                raise
            except Exception as exc:
                raise EvalError(f"build_family() raised: {exc}") from exc
            return normalizer(raw)
        finally:
            sys.modules.pop(module_name, None)


def _read_model_source(candidate_dir: Path) -> bytes:
    path = candidate_dir / "model.py"
    if not path.is_file():
        raise EvalError("candidate is missing model.py")
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"could not read model.py: {exc}") from exc
    if len(source) > _MODEL_MAX_BYTES:
        raise EvalError(f"model.py is {len(source)} bytes; limit is {_MODEL_MAX_BYTES}")
    return source


def _budget_failure(checked: int, required: int) -> EvalError:
    return EvalError(
        "closure verification operation budget exhausted after "
        f"{checked} of {required} unordered pairs"
    )


def _check_union_closed_array(
    sets: tuple[int, ...],
    n: int,
    pair_budget: int,
) -> None:
    """Check every unordered pair with a dense direct-address numpy table."""

    masks = _NP_ARRAY(sets, dtype=_NP_UINT32)
    present = _NP_ZEROS(1 << n, dtype=_NP_BOOL)
    present[masks] = True
    required = len(sets) * (len(sets) + 1) // 2
    checked = 0
    for left_index, left in enumerate(sets):
        row_size = len(sets) - left_index
        if checked + row_size > pair_budget:
            raise _budget_failure(checked, required)
        rights = masks[left_index:]
        unions = _BITWISE_OR(_NP_UINT32(left), rights)
        missing = _FLATNONZERO(_LOGICAL_NOT(present[unions]))
        if missing.size:
            right_index = left_index + int(missing[0])
            right = sets[right_index]
            union = left | right
            raise EvalError(
                f"union closure failed: {left:#x} | {right:#x} = {union:#x} "
                "is not a member"
            )
        checked += row_size


def _check_union_closed_direct(
    sets: tuple[int, ...],
    n: int,
    pair_budget: int,
) -> None:
    """Independently check every unordered pair with a bytearray table."""

    present = bytearray(1 << n)
    for mask in sets:
        present[mask] = 1
    required = len(sets) * (len(sets) + 1) // 2
    checked = 0
    for left_index, left in enumerate(sets):
        row_size = len(sets) - left_index
        if checked + row_size > pair_budget:
            raise _budget_failure(checked, required)
        for right in sets[left_index:]:
            union = left | right
            if not present[union]:
                raise EvalError(
                    f"independent union check failed: {left:#x} | {right:#x} = "
                    f"{union:#x} is not a member"
                )
        checked += row_size


def _measure_bitwalk(certificate: _Certificate) -> _Measurement:
    frequencies = [0] * certificate.n
    used_mask = 0
    for mask in certificate.sets:
        used_mask |= mask
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            frequencies[bit.bit_length() - 1] += 1
            remaining ^= bit
    return _Measurement(
        frequencies=tuple(frequencies),
        max_freq=max(frequencies),
        used_elements=used_mask.bit_count(),
    )


def _measure_scan(certificate: _Certificate) -> _Measurement:
    frequencies = tuple(
        sum(1 for mask in certificate.sets if mask & (1 << element))
        for element in range(certificate.n)
    )
    used_mask = 0
    for mask in certificate.sets:
        used_mask |= mask
    return _Measurement(
        frequencies=frequencies,
        max_freq=max(frequencies),
        used_elements=sum(1 for element in range(certificate.n) if used_mask & (1 << element)),
    )


def _verify_primary(certificate: _Certificate) -> _Measurement:
    _check_union_closed_array(
        certificate.sets,
        certificate.n,
        _CLOSURE_PAIR_BUDGET,
    )
    return _measure_bitwalk(certificate)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Verify one exact union-closed certificate and return measured metrics."""

    if CONFIG_ERROR is not None:
        raise EvalError(f"cell configuration is invalid: {CONFIG_ERROR}")
    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    source = _read_model_source(candidate_dir)
    replay_count = 2 if stage == 1 else 1
    candidate_window = STAGES[stage].timeout_s * 0.75 / replay_count

    first = _extract_certificate(
        source,
        _MONOTONIC() + candidate_window,
        f"stage{stage}_first",
        _normalize_certificate,
    )
    first_measurement = _verify_primary(first)
    if stage == 1:
        replay = _extract_certificate(
            source,
            _MONOTONIC() + candidate_window,
            "stage1_replay",
            _normalize_certificate_replay,
        )
        if replay.wire != first.wire:
            raise EvalError("replay produced different normalized certificate bytes")
        replay_measurement = _verify_primary(replay)
        if replay_measurement != first_measurement:
            raise EvalError("replay produced different exact measurements")
        _check_union_closed_direct(
            first.sets,
            first.n,
            _CLOSURE_PAIR_BUDGET,
        )
        independent_measurement = _measure_scan(first)
        if independent_measurement != first_measurement:
            raise EvalError("independent frequency or ground-set cross-check disagreed")
    return first_measurement.metrics(first)


def ceiling() -> dict[str, float | str] | None:
    """Return no proved optimization ceiling for this open problem."""

    return None


DESCRIPTORS = [
    {
        "name": "ground_set_used",
        "metric": "ground_set_used",
        "bins": 12,
        "lo": 0.5,
        "hi": 24.5,
    },
    {
        "name": "family_size",
        "metric": "family_size",
        "bins": 10,
        "lo": 1.0,
        "hi": 20000.5,
    },
]
