from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "vrp"
PACK = CAMPAIGN / "evaluators" / "vrp"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str = "tiny-12-validation",
    name: str = "test_vrp_evaluator",
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    path = PACK / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _hand_instance(
    evaluator: ModuleType,
    *,
    capacity: int = 10,
    first_earliest: int = 10,
    first_latest: int = 20,
) -> object:
    text = f"""HAND_GENERATED_GATE_FIXTURE

VEHICLE
NUMBER     CAPACITY
5          {capacity}

CUSTOMER
CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME

0 0 0 0 0 100 0
1 1 0 3 {first_earliest} {first_latest} 5
2 2 0 3 0 30 0
"""
    return evaluator.parse_solomon_text(text, "hand-fixture")


def test_generated_fixture_parser_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_parser_roundtrip")
    fixtures = (
        (12, evaluator._TINY_SEED, "generated-tiny-12.txt"),
        (100, evaluator._GENERATED_100_SEED, "generated-100.txt"),
    )
    for customer_count, seed, filename in fixtures:
        fixture = PACK / "fixtures" / filename
        source = fixture.read_text(encoding="utf-8")
        name = f"AUTOEVOLVE_GENERATED_{customer_count}_SEED_{seed}"

        assert source == evaluator.generate_fixture_text(customer_count, seed, name)
        parsed = evaluator.parse_solomon_text(source, fixture.name)
        serialized = evaluator.format_solomon_instance(parsed)

        assert evaluator.parse_solomon_text(serialized, "roundtrip") == parsed
        assert parsed.customer_count == customer_count
        assert parsed.depot.customer_id == 0


def test_objective_key_is_vehicle_count_then_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_objective")
    fewer_vehicles = evaluator.Measurement(1, 500.0, 2.0)
    shorter_with_more_vehicles = evaluator.Measurement(2, 1.0, 1.0)

    assert fewer_vehicles.objective < shorter_with_more_vehicles.objective


def test_capacity_violation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_capacity")
    instance = _hand_instance(evaluator, capacity=5)
    solution = evaluator.Solution(routes=((0, 1, 2, 0),))

    with pytest.raises(EvalError, match="exceeds capacity"):
        evaluator._verify_solution(solution, instance)


def test_late_arrival_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_late")
    instance = _hand_instance(evaluator, first_earliest=0, first_latest=0)
    solution = evaluator.Solution(routes=((0, 1, 2, 0),))

    with pytest.raises(EvalError, match="after window closes"):
        evaluator._verify_solution(solution, instance)


def test_missing_customer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_missing")
    instance = _hand_instance(evaluator)
    solution = evaluator.Solution(routes=((0, 1, 0),))

    with pytest.raises(EvalError, match="misses customers: 2"):
        evaluator._verify_solution(solution, instance)


def test_duplicate_customer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_duplicate")
    instance = _hand_instance(evaluator)
    solution = evaluator.Solution(routes=((0, 1, 1, 2, 0),))

    with pytest.raises(EvalError, match="customer 1 is visited more than once"):
        evaluator._verify_solution(solution, instance)


def test_waiting_for_window_open_is_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_waiting")
    instance = _hand_instance(evaluator)
    solution = evaluator.Solution(routes=((0, 1, 2, 0),))

    measured = evaluator._verify_solution(solution, instance)

    assert measured.vehicle_count == 1
    assert measured.total_distance == 4.0
    assert measured.mean_route_customers == 2.0


def test_depot_in_customer_position_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_depot_position")
    instance = _hand_instance(evaluator)
    solution = evaluator.Solution(routes=((0, 1, 0, 2, 0),))

    with pytest.raises(EvalError, match="contains depot 0 as a customer"):
        evaluator._verify_solution(solution, instance)


def test_seed_passes_and_descriptor_metrics_are_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_seed")

    scores = evaluator.evaluate(PACK / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "total_distance"
    assert evaluator.MAXIMIZE is False
    assert scores[evaluator.METRIC] > 0.0
    assert evaluator.DESCRIPTORS
    assert len(evaluator.DESCRIPTORS) == 2
    for descriptor in evaluator.DESCRIPTORS:
        assert descriptor["name"] in scores
        assert descriptor["metric"] in scores


def test_solver_signature_supports_legacy_and_seeded_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_signature")
    calls: list[tuple[object, ...]] = []

    def legacy(instance: object) -> dict[str, object]:
        calls.append((instance,))
        return {"routes": []}

    def modern(
        instance: object,
        deadline: float | None = None,
        seed: int = 0,
    ) -> dict[str, object]:
        calls.append((instance, deadline, seed))
        return {"routes": []}

    payload = {"probe": True}
    evaluator._call_solver(legacy, payload, 12.5, 7)
    evaluator._call_solver(modern, payload, 12.5, 7)

    assert calls == [(payload,), (payload, 12.5, 7)]


def test_campaign_empty_bounds_timeouts_markers_and_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_vrp_contract")
    campaign = load_campaign(CAMPAIGN)
    bounds = load_bounds(CAMPAIGN)

    assert campaign.name == "vrp"
    assert campaign.evaluator_path == PACK.resolve()
    assert [cell.key for cell in campaign.cells[:2]] == [
        "tiny-12-validation",
        "generated-100",
    ]
    assert all(cell.target is None for cell in campaign.cells)
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert bounds == ()
    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert all(
        cell.timeout_s == 300.0
        for key, cell in evaluator._CELLS.items()
        if key.endswith("-frontier")
    )

    baseline_source = (PACK / "baseline" / "solver.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    evaluator_spec = (PACK / "spec.md").read_text(encoding="utf-8")
    campaign_spec = (CAMPAIGN / "spec.md").read_text(encoding="utf-8")
    assert evaluator.DISTANCE_CONVENTION in evaluator_spec
    assert "minimize vehicle count first" in evaluator_spec.lower()
    assert "no network code" in evaluator_spec.lower()
    assert "published best-known distance" in campaign_spec
    assert evaluator.ceiling() is None
