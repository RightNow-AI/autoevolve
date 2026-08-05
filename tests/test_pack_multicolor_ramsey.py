from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "multicolor-ramsey"
EVALUATOR = PACK / "evaluators" / "multiramsey"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    name: str,
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    return _load(EVALUATOR / "evaluate.py", name)


@pytest.mark.parametrize("cell", ["n5-validation", "n49-frontier"])
def test_seed_passes_stage_zero_gate_and_reports_metric(
    cell: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, cell, f"test_multiramsey_seed_{cell}")

    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "n_vertices"
    assert scores[evaluator.METRIC] == 5.0
    assert scores["red_edges"] == 5.0
    assert scores["blue_edges"] == 5.0
    assert scores["green_edges"] == 0.0
    assert scores["yellow_edges"] == 0.0
    assert evaluator.MAXIMIZE is True


def test_committed_blue_k4_mutant_fails_with_named_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n5-validation",
        "test_multiramsey_blue_k4",
    )
    mutant = EVALUATOR / "fixtures" / "mutants" / "blue_k4"

    with pytest.raises(EvalError, match=r"BLUE contains K4"):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_container_with_varying_reads_cannot_fool_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n5-validation",
        "test_multiramsey_hostile",
    )
    hostile = EVALUATOR / "fixtures" / "mutants" / "hostile_container"

    with pytest.raises(EvalError, match=r"BLUE contains K4"):
        evaluator.evaluate(hostile, stage=0)


def test_boolean_integer_fields_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n5-validation",
        "test_multiramsey_bool_rejected",
    )

    with pytest.raises(EvalError, match=r"n must be an integer, got bool"):
        evaluator._normalize_certificate({"n": True, "edge_colors": []})


def test_campaign_and_all_seven_field_bounds_parse_with_repo_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "multicolor-ramsey"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "n5-validation",
        "n49-frontier",
    ]
    assert [cell.target for cell in campaign.cells] == [5.0, 49.0]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 2
    for bound in bounds:
        assert bound.claim
        assert bound.value
        assert bound.direction
        assert bound.who_and_year
        assert bound.source_url == "https://arxiv.org/abs/2509.03784"
        assert bound.checked_on == "2026-08-05"
        assert bound.how_to_recheck


def test_contract_returns_two_structural_descriptor_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n49-frontier",
        "test_multiramsey_descriptors",
    )

    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "red_density",
        "distinct_color_class_sizes",
    }
    assert evaluator.ceiling()["value"] == 74.0


def test_every_single_edge_increment_matches_full_recount() -> None:
    search = _load(PACK / "search.py", "test_multiramsey_search_delta")
    n = 6
    colorings = [
        [index % 4 for index in range(n * (n - 1) // 2)],
        [(index * 3 + 1) % 4 for index in range(n * (n - 1) // 2)],
    ]

    for colors in colorings:
        state = search.state_from_colors(n, colors)
        for edge in combinations(range(n), 2):
            old_color = search.edge_color(state.colors, n, *edge)
            for new_color in range(4):
                if new_color == old_color:
                    continue
                candidate = search.copy_state(state)
                move = search.measure_move(candidate, edge, new_color)
                search.apply_move(candidate, move)
                assert tuple(candidate.counts) == search.full_recount(n, candidate.colors)


def test_verified_writer_refuses_incremental_recount_disagreement(
    tmp_path: Path,
) -> None:
    search = _load(PACK / "search.py", "test_multiramsey_verified_writer")
    n = 5
    colors = [1] * (n * (n - 1) // 2)
    target = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="disagreed with the full recount"):
        search.write_verified_certificate(target, n, colors, (0, 0, 0, 0))
    assert not target.exists()
