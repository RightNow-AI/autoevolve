"""Exact Euclidean scorer for the bundled routing heuristic task."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [
    StageSpec(name="small-instances", timeout_s=20.0),
    StageSpec(name="all-instances", timeout_s=60.0),
]
GATE: str = "valid"

PACK_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = PACK_DIR / "fixtures"
Point = tuple[float, float]
Instance = tuple[str, list[Point]]


def _load_module(candidate_dir: Path) -> ModuleType:
    entry_path = candidate_dir / "heuristic.py"
    if not entry_path.is_file():
        raise EvalError(f"candidate is missing {entry_path.name}")
    module_name = f"_autoevolve_routing_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_instances() -> list[Instance]:
    raw = json.loads((FIXTURE_DIR / "instances.json").read_text(encoding="utf-8"))
    instances: list[Instance] = []
    for item in raw["instances"]:
        points = [(float(point[0]), float(point[1])) for point in item["points"]]
        instances.append((str(item["name"]), points))
    return instances


def _stage_instances(stage: int) -> list[Instance]:
    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    instances = _load_instances()
    return instances[:3] if stage == 0 else instances


def _validated_tour(module: ModuleType, name: str, points: list[Point]) -> list[int]:
    try:
        raw_tour = module.build_tour(points[:])
    except Exception as exc:
        raise EvalError(f"instance {name} raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw_tour, list):
        raise EvalError(f"instance {name} returned a non-list tour")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_tour):
        raise EvalError(f"instance {name} returned a non-integer tour index")
    if len(raw_tour) != len(points) or sorted(raw_tour) != list(range(len(points))):
        raise EvalError(f"instance {name} tour is not a permutation of range({len(points)})")
    return raw_tour


def _tour_cost(points: list[Point], tour: list[int]) -> float:
    cost = 0.0
    for position, point_index in enumerate(tour):
        next_index = tour[(position + 1) % len(tour)]
        x1, y1 = points[point_index]
        x2, y2 = points[next_index]
        cost += math.hypot(x2 - x1, y2 - y1)
    return cost


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Validate every tour before returning exact aggregate cycle costs."""
    instances = _stage_instances(stage)
    candidate = _load_module(candidate_dir)
    costs: list[float] = []
    for name, points in instances:
        tour = _validated_tour(candidate, name, points)
        costs.append(_tour_cost(points, tour))
    total = sum(costs)
    return {
        GATE: 1.0,
        "tour_cost": total,
        "mean_cost": total / len(costs),
        **source_metrics(candidate_dir, "heuristic.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no ceiling because exact optimal tour costs are not computed."""
    return None

# Primary metric declaration consumed by the engine when locking a contract.
METRIC = "tour_cost"
MAXIMIZE = False

# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
