from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "routing-heuristic"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _independent_cycle_cost(
    points: list[tuple[float, float]],
    tour: list[int],
) -> float:
    total = 0.0
    for position, index in enumerate(tour):
        next_index = tour[(position + 1) % len(tour)]
        total += math.dist(points[index], points[next_index])
    return total


def test_baseline_tours_are_valid_and_cost_matches_independent_scorer() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_routing_evaluator")
    baseline = _load_module(PACK / "baseline" / "heuristic.py", "test_routing_baseline")
    fixture = json.loads(
        (PACK / "fixtures" / "instances.json").read_text(encoding="utf-8")
    )
    independent_total = 0.0
    for instance in fixture["instances"]:
        points = [(float(x), float(y)) for x, y in instance["points"]]
        tour = baseline.build_tour(points)
        assert len(tour) == len(points)
        assert sorted(tour) == list(range(len(points)))
        independent_total += _independent_cycle_cost(points, tour)
    scores = evaluator.evaluate(PACK / "baseline", stage=1)
    assert scores["valid"] == 1.0
    assert scores["tour_cost"] == pytest.approx(independent_total, abs=1e-9)
    assert scores["mean_cost"] == pytest.approx(independent_total / 8.0, abs=1e-9)


def test_truncated_tour_mutant_fails_gate() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_routing_truncated")
    mutant = PACK / "fixtures" / "mutants" / "truncated_tour"
    with pytest.raises(EvalError, match="uniform-30"):
        evaluator.evaluate(mutant, stage=0)


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_routing_fixture_generator",
    )
    generator.write_fixtures(tmp_path)
    committed = (PACK / "fixtures" / "instances.json").read_bytes()
    assert (tmp_path / "instances.json").read_bytes() == committed

