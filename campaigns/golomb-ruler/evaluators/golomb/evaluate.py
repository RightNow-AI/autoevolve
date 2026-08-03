"""Golomb ruler evaluator: exact certificate, integer arithmetic only.

A Golomb ruler of order n is a set of n integer marks whose pairwise
differences are all distinct. Optimality is proven only through order 28, so
shorter rulers at higher orders are a real computational contribution.

The gate is exact and total: every pairwise difference is compared, in
integers, with no tolerance anywhere. The candidate only produces marks; this
module re-derives every quantity that reaches a metric.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [StageSpec(name="verify", timeout_s=60.0)]
GATE = "golomb"
METRIC = "length"
MAXIMIZE = False

# Configuration is read at import time, BEFORE any candidate code is loaded,
# so a candidate cannot choose the order it is judged against.
_CELL = os.environ.get("AUTOEVOLVE_CELL", "order-11")
try:
    ORDER = int(_CELL.rsplit("-", 1)[1])
except (IndexError, ValueError) as _exc:  # pragma: no cover - configuration error
    raise EvalError(f"AUTOEVOLVE_CELL must look like order-<n>, got {_CELL!r}") from _exc


def _normalize_marks(raw: object) -> tuple[int, ...]:
    """Read the candidate's marks exactly once into immutable integers.

    A returned container may be a subclass whose __getitem__ or __iter__ is
    candidate code answering differently on each read, so every later clause
    reads only this snapshot. bool is rejected explicitly because it is a
    subclass of int.
    """

    try:
        items = list(raw)
    except TypeError as exc:
        raise EvalError(
            f"build() must return an iterable of ints, got {type(raw).__name__}"
        ) from exc
    marks: list[int] = []
    for index, item in enumerate(items):
        if type(item) is not int:
            raise EvalError(
                f"mark {index} must be a plain int, got {type(item).__name__}"
            )
        marks.append(item)
    return tuple(marks)


def _check_golomb(marks: tuple[int, ...]) -> int:
    """Verify the Golomb property exactly and return the distinct difference count."""

    if len(marks) != ORDER:
        raise EvalError(f"ruler has order {len(marks)}, the cell requires order {ORDER}")
    if marks[0] != 0:
        raise EvalError(f"first mark must be 0, got {marks[0]}")
    for i in range(1, len(marks)):
        if marks[i] <= marks[i - 1]:
            raise EvalError(f"marks must strictly increase; mark {i} is {marks[i]}")
    seen: dict[int, tuple[int, int]] = {}
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            diff = marks[j] - marks[i]
            if diff in seen:
                a, b = seen[diff]
                raise EvalError(
                    f"difference {diff} repeats: marks[{a}],marks[{b}] and marks[{i}],marks[{j}]"
                )
            seen[diff] = (i, j)
    return len(seen)


def _load_candidate(candidate_dir: Path) -> object:
    path = candidate_dir / "ruler.py"
    if not path.is_file():
        raise EvalError("candidate is missing ruler.py")
    spec = importlib.util.spec_from_file_location("golomb_candidate", path)
    if spec is None or spec.loader is None:
        raise EvalError("could not load ruler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["golomb_candidate"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"ruler.py failed to import: {exc}") from exc
    return module


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Verify the Golomb property exactly, then score the measured length."""

    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    module = _load_candidate(candidate_dir)
    build = getattr(module, "build", None)
    if not callable(build):
        raise EvalError("ruler.py must define callable build()")
    try:
        raw = build(ORDER)
    except Exception as exc:
        raise EvalError(f"build() raised: {exc}") from exc
    marks = _normalize_marks(raw)
    distinct = _check_golomb(marks)
    return {
        GATE: 1.0,
        "length": float(marks[-1]),
        "order": float(len(marks)),
        "distinct_differences": float(distinct),
    }


def ceiling() -> dict[str, float | str] | None:
    """No certified ceiling. A lower bound on length is not computed here."""
    return None
