from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "ramsey-lower-bound"
EVALUATOR = PACK / "evaluators" / "ramsey"


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


@pytest.mark.parametrize(
    ("cell", "expected_n"),
    [
        ("k3-smoke", 4.0),
        ("k4-climb", 17.0),
        ("k5-frontier", 37.0),
    ],
)
def test_seed_passes_exact_stage_zero_gate_and_reports_metric(
    cell: str,
    expected_n: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, cell, f"test_ramsey_seed_{cell}")

    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "n_vertices"
    assert scores[evaluator.METRIC] == expected_n
    assert scores["target_clique"] == float(evaluator.S)


def test_monochromatic_mutant_fails_with_named_blue_clique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "k4-climb", "test_ramsey_monochromatic")
    mutant = EVALUATOR / "fixtures" / "mutants" / "monochromatic"

    with pytest.raises(EvalError, match=r"BLUE contains K4"):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_mapping_cannot_change_the_certificate_between_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "k3-smoke", "test_ramsey_hostile_gate")
    hostile = EVALUATOR / "fixtures" / "mutants" / "hostile_container"

    with pytest.raises(EvalError, match=r"BLUE contains K3"):
        evaluator.evaluate(hostile, stage=0)


def test_bounded_bitset_gate_fails_closed_when_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "k3-smoke", "test_ramsey_gate_budget")
    certificate = evaluator._normalize_certificate(
        {"form": "circulant", "n": 4, "red_diffs": [1]}
    )
    red, blue = evaluator._expand_masks(certificate)

    with pytest.raises(EvalError, match="operation budget exhausted"):
        evaluator._bitset_verdict(red, blue, evaluator.S, operation_limit=0)


def test_campaign_and_all_seven_field_bounds_parse_with_repo_loaders() -> None:
    config = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert config.name == "ramsey-lower-bound"
    assert config.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in config.cells] == [
        "k3-smoke",
        "k4-climb",
        "k5-frontier",
    ]
    assert len(bounds) == 3
    for bound in bounds:
        assert bound.claim
        assert bound.value
        assert bound.direction
        assert bound.who_and_year
        assert bound.source_url
        assert bound.checked_on == "2026-08-03"
        assert bound.how_to_recheck


def test_contract_declares_two_structural_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "k5-frontier", "test_ramsey_descriptors")

    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "red_density",
        "is_circulant",
    }
    assert evaluator.MAXIMIZE is True
    assert evaluator.ceiling()["value"] == 45.0
