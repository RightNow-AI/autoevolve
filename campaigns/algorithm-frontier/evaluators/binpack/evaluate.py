"""Exact scorer for one-dimensional bin-packing candidates."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [StageSpec(name="exact-binpack", timeout_s=15.0)]
GATE = "valid"
PACK_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = PACK_DIR / "fixtures"


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "model.py"
    if not path.is_file():
        raise EvalError("candidate is missing model.py")
    spec = importlib.util.spec_from_file_location(
        f"_autoevolve_binpack_{abs(hash(path.resolve()))}",
        path,
    )
    if spec is None or spec.loader is None:
        raise EvalError("candidate model.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_family() -> dict[str, Any]:
    cell = os.environ.get("AUTOEVOLVE_CELL", "uniform")
    if cell not in {"uniform", "clustered"}:
        raise EvalError("AUTOEVOLVE_CELL must be uniform or clustered")
    return json.loads((FIXTURE_DIR / f"{cell}.json").read_text(encoding="utf-8"))


def _validate_bins(
    bins: object,
    items: list[int],
    capacity: int,
    instance_key: str,
) -> int:
    if not isinstance(bins, list) or not bins:
        raise EvalError(f"{instance_key} returned no bins")
    placed: list[int] = []
    for bin_index, indexes in enumerate(bins):
        if not isinstance(indexes, list) or not indexes:
            raise EvalError(f"{instance_key} bin {bin_index} is not a non-empty list")
        load = 0
        for item_index in indexes:
            if isinstance(item_index, bool) or not isinstance(item_index, int):
                raise EvalError(f"{instance_key} returned a non-integer item index")
            if item_index < 0 or item_index >= len(items):
                raise EvalError(f"{instance_key} returned item index {item_index} out of range")
            placed.append(item_index)
            load += items[item_index]
        if load > capacity:
            raise EvalError(
                f"{instance_key} bin {bin_index} exceeds capacity: {load} > {capacity}"
            )
    expected = list(range(len(items)))
    if sorted(placed) != expected:
        raise EvalError(f"{instance_key} must place every item exactly once")
    return len(bins)


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Validate every placement and minimize the exact total bin count."""

    if stage != 0:
        raise EvalError(f"unknown stage {stage}")
    candidate = _load_candidate(candidate_dir)
    pack = getattr(candidate, "pack", None)
    if not callable(pack):
        raise EvalError("candidate must define callable pack(items, capacity)")
    payload = _load_family()
    capacity = int(payload["capacity"])
    total_bins = 0
    for instance in payload["instances"]:
        key = str(instance["key"])
        items = [int(value) for value in instance["items"]]
        try:
            bins = pack(list(items), capacity)
        except Exception as exc:
            raise EvalError(f"{key} candidate execution failed: {exc}") from exc
        total_bins += _validate_bins(bins, items, capacity, key)
    return {GATE: 1.0, "bins_used": float(total_bins)}


def ceiling() -> dict[str, float | str] | None:
    """Return no aggregate lower bound for the committed family."""

    return None

