from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "circle-packing"
EVALUATOR = PACK / "evaluators" / "circlepack"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str = "n2-validation",
    name: str = "test_circlepack_evaluator",
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    path = EVALUATOR / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(tmp_path: Path, source: str, name: str) -> Path:
    candidate = tmp_path / name
    candidate.mkdir()
    (candidate / "solver.py").write_text(source, encoding="utf-8")
    return candidate


def test_validation_cell_optimum_is_derived_from_unit_square_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_optimum")
    candidate = _candidate(
        tmp_path,
        """def solve(n):
    return [(float(bit), float(bit)) for bit in range(n)]
""",
        "diagonal",
    )

    scores = evaluator.evaluate(candidate)

    # For any two points in the unit square, each coordinate difference is at
    # most one. Therefore squared distance is at most 1**2 + 1**2. Opposite
    # corners attain that upper bound, so this derives the validation optimum.
    coordinate_span = 1.0
    squared_upper_bound = coordinate_span**2 + coordinate_span**2
    expected_optimum = math.sqrt(squared_upper_bound)
    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == expected_optimum
    assert scores["circle_radius"] == expected_optimum / (2.0 * (1.0 + expected_optimum))
    assert evaluator.METRIC == "min_pairwise_distance"
    assert evaluator.MAXIMIZE is True


def test_point_outside_square_raises_eval_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_outside")
    candidate = _candidate(
        tmp_path,
        """def solve(n):
    return [(-0.25, 0.5), (1.0, 1.0)]
""",
        "outside",
    )

    with pytest.raises(EvalError, match="outside the closed unit square"):
        evaluator.evaluate(candidate)


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("too-few", "return [(0.5, 0.5)]"),
        ("too-many", "return [(0.5, 0.5) for _ in range(n + 1)]"),
    ],
)
def test_wrong_point_count_raises_eval_error(
    name: str,
    body: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_count")
    candidate = _candidate(
        tmp_path,
        f"def solve(n):\n    {body}\n",
        name,
    )

    with pytest.raises(EvalError, match="exactly 2 points"):
        evaluator.evaluate(candidate)


def test_deadline_and_seed_are_passed_when_declared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_compute_contract")
    candidate = _candidate(
        tmp_path,
        """def solve(n, *, deadline, seed):
    if deadline <= 0.0 or seed <= 0:
        raise ValueError("missing compute controls")
    return [(float(bit), float(bit)) for bit in range(n)]
""",
        "compute-contract",
    )

    scores = evaluator.evaluate(candidate)

    assert scores[evaluator.GATE] == 1.0


def test_non_finite_coordinate_raises_eval_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_non_finite")
    candidate = _candidate(
        tmp_path,
        """def solve(n):
    return [(0.0, 0.0), (float("nan"), 1.0)]
""",
        "non-finite",
    )

    with pytest.raises(EvalError, match="must be finite"):
        evaluator.evaluate(candidate)


def test_coincident_points_score_zero_and_return_both_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_coincident")
    candidate = _candidate(
        tmp_path,
        """def solve(n):
    return [(0.5, 0.5) for _ in range(n)]
""",
        "coincident",
    )

    scores = evaluator.evaluate(candidate)

    assert scores[evaluator.METRIC] == 0.0
    assert len(evaluator.DESCRIPTORS) == 2
    assert evaluator.DESCRIPTORS
    for descriptor in evaluator.DESCRIPTORS:
        assert descriptor["name"] in scores
        assert descriptor["metric"] in scores


def test_seed_passes_a_shortened_validation_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator(monkeypatch, name="test_circlepack_seed")
    assert evaluator.STAGES[0].timeout_s == 15.0
    evaluator.STAGES = [
        StageSpec(name="search-and-exact-geometry-gate", timeout_s=0.08)
    ]
    evaluator._DEADLINE_HEADROOM_S = 0.02

    scores = evaluator.evaluate(EVALUATOR / "baseline")

    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] >= 0.0
    assert scores["point_count"] == 2.0


def test_campaign_shape_empty_bounds_and_baseline_fence() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "circle-packing"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "n2-validation",
        "n10-calibration",
        "n20-calibration",
        "n30-calibration",
        "n31-frontier",
        "n37-frontier",
        "n43-frontier",
        "n51-frontier",
        "n62-frontier",
    ]
    assert "calibration" in campaign.ladder
    assert all(cell.target is None for cell in campaign.cells)
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert bounds == ()
    assert json.loads((PACK / "bounds.json").read_text(encoding="utf-8")) == {"bounds": []}

    baseline = (EVALUATOR / "baseline" / "solver.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline
    assert "# EVOLVE-BLOCK-END" in baseline


@pytest.mark.parametrize(
    "cell",
    ["n10-calibration", "n20-calibration", "n30-calibration"],
)
def test_calibration_cells_are_explicit_and_receive_full_search_timeout(
    cell: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        cell=cell,
        name=f"test_circlepack_{cell.replace('-', '_')}",
    )

    assert "calibration" in cell
    assert evaluator.STAGES == [
        StageSpec(name="search-and-exact-geometry-gate", timeout_s=300.0)
    ]
    assert evaluator.DESCRIPTORS[0]["hi"] >= evaluator.POINT_COUNT


@pytest.mark.parametrize(
    "cell",
    ["n31-frontier", "n37-frontier", "n43-frontier", "n51-frontier", "n62-frontier"],
)
def test_frontier_cells_receive_full_search_timeout(
    cell: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        cell=cell,
        name=f"test_circlepack_{cell.replace('-', '_')}",
    )

    assert evaluator.STAGES == [
        StageSpec(name="search-and-exact-geometry-gate", timeout_s=300.0)
    ]
    assert evaluator.DESCRIPTORS[0]["hi"] >= evaluator.POINT_COUNT
