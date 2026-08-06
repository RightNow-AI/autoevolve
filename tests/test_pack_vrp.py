from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from dataclasses import astuple
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec
from campaigns.vrp.bounds_parser import parse_sintef_page
from campaigns.vrp.objective import decode_objective_value, is_better_result

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


def _write_singleton_solver(candidate_dir: Path) -> None:
    candidate_dir.mkdir()
    (candidate_dir / "solver.py").write_text(
        """def solve(instance, deadline=None, seed=0):
    del deadline, seed
    return {"routes": [[0, int(customer["id"]), 0] for customer in instance["customers"]]}
""",
        encoding="utf-8",
    )


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
    assert fewer_vehicles.is_better_than(shorter_with_more_vehicles)


def test_hierarchical_comparison_uses_vehicles_then_distance() -> None:
    assert is_better_result((1, 500.0), (2, 1.0))
    assert not is_better_result((2, 1.0), (1, 500.0))
    assert is_better_result((2, 99.99), (2, 100.0))
    assert not is_better_result((2, 100.0), (2, 100.0))


def test_file_cell_matches_equivalent_named_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    _write_singleton_solver(candidate)
    named = _load_evaluator(monkeypatch, name="test_vrp_named_tiny")
    named_scores = named.evaluate(candidate)

    file_cell = _load_evaluator(
        monkeypatch,
        cell="file:generated-tiny-12.txt",
        name="test_vrp_file_tiny",
    )
    file_scores = file_cell.evaluate(candidate)

    assert file_cell.CELL.key == "file:generated-tiny-12.txt"
    assert file_cell.CELL.fixture == "generated-tiny-12.txt"
    assert file_cell.CELL.timeout_s == 300.0
    # Compared field by field rather than with ==. The evaluator is imported
    # twice under different module names, so each import defines its own
    # Instance and Stop classes and dataclass equality is False between them
    # however identical the data is.
    assert file_cell.INSTANCE.name == named.INSTANCE.name
    assert file_cell.INSTANCE.vehicle_limit == named.INSTANCE.vehicle_limit
    assert file_cell.INSTANCE.capacity == named.INSTANCE.capacity
    assert [astuple(stop) for stop in file_cell.INSTANCE.stops] == [
        astuple(stop) for stop in named.INSTANCE.stops
    ]
    assert file_scores == named_scores


def test_file_cell_timeout_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_VRP_TIMEOUT_S", "123.5")
    evaluator = _load_evaluator(
        monkeypatch,
        cell="file:generated-tiny-12.txt",
        name="test_vrp_file_timeout",
    )

    assert evaluator.CELL.timeout_s == 123.5
    assert evaluator.STAGES[0].timeout_s == 123.5


@pytest.mark.parametrize(
    "cell",
    [
        "file:../generated-tiny-12.txt",
        f"file:{(PACK / 'fixtures' / 'generated-tiny-12.txt').resolve()}",
    ],
)
def test_file_cell_rejects_unsafe_paths(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
) -> None:
    with pytest.raises(EvalError, match="relative|contain|escapes"):
        _load_evaluator(monkeypatch, cell=cell, name=f"test_vrp_unsafe_{len(cell)}")


def test_file_cell_seed_is_stable_across_python_hash_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        cell="file:generated-tiny-12.txt",
        name="test_vrp_stable_seed_source",
    )
    assert "hash(" not in inspect.getsource(evaluator._stable_path_seed)

    code = """
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("seed_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules["seed_probe"] = module
spec.loader.exec_module(module)
print(module.CELL.seed)
"""
    seeds: list[int] = []
    for hash_seed in ("1", "987654"):
        env = dict(os.environ)
        env["AUTOEVOLVE_CELL"] = "file:generated-tiny-12.txt"
        env["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", code, str(PACK / "evaluate.py")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        seeds.append(int(completed.stdout.strip()))
    assert seeds[0] == seeds[1] == evaluator.CELL.seed


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


def test_campaign_bounds_timeouts_markers_and_specs(
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
    assert all(bound.direction == "lexicographic_lower_is_better" for bound in bounds)
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


def test_sintef_table_parser_writes_exact_bounds_schema() -> None:
    html = """
<table>
  <tr><th>Instance</th><th>Vehicles</th><th>Distance</th><th>Reference</th><th>Date</th></tr>
  <tr><td>C101</td><td>10</td><td>828.94</td><td>KDMSS</td><td>1999</td></tr>
  <tr><td>R1_2_1</td><td>20</td><td>999.01</td><td>ABC</td><td>2025</td></tr>
</table>
"""
    parsed = parse_sintef_page(
        html,
        "https://www.sintef.no/projectweb/top/vrptw/100-customers/",
        "2026-08-06",
    )

    assert parsed.row_errors == ()
    assert len(parsed.bounds) == 2
    expected_fields = {
        "claim",
        "value",
        "direction",
        "who_and_year",
        "source_url",
        "checked_on",
        "how_to_recheck",
    }
    assert set(parsed.bounds[0]) == expected_fields
    assert decode_objective_value(parsed.bounds[0]["value"]) == (10, 828.94)
    assert parsed.bounds[0]["who_and_year"] == "KDMSS; 1999"


def test_modal_portfolio_is_parallel_crash_tolerant_and_uses_normal_contract() -> None:
    source = (CAMPAIGN / "modal_portfolio.py").read_text(encoding="utf-8")

    assert "single_use_containers=True" in source
    assert "run_instance.map(jobs, return_exceptions=True)" in source
    assert "run_cascade(load_evaluator(Path(sys.argv[1])), Path(sys.argv[2]))" in source
    assert '"uv",' in source
    assert '"--frozen",' in source
    assert 'modal.Volume.from_name("autoevolve-store"' in source


def test_fetch_bounds_contains_only_source_pages_not_best_known_constants() -> None:
    source = (CAMPAIGN / "fetch_bounds.py").read_text(encoding="utf-8")

    assert source.count("https://www.sintef.no/projectweb/top/vrptw/") == 6
    assert "urllib.request.urlopen" in source
    bounds_payload = json.loads((CAMPAIGN / "bounds.json").read_text(encoding="utf-8"))
    assert set(bounds_payload) == {"bounds"}
    assert isinstance(bounds_payload["bounds"], list)
