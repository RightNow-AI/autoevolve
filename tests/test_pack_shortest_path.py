from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "shortest-path"
PACK = CAMPAIGN / "evaluators" / "shortest_path"


def _load_module(path: Path, name: str) -> ModuleType:
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


def _load_evaluator(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", "small-validation")
    return _load_module(PACK / "evaluate.py", name)


def test_seed_passes_exact_gate_and_reports_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_seed")

    scores = evaluator.evaluate(PACK / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "queries_per_second"
    assert evaluator.MAXIMIZE is True
    assert scores[evaluator.METRIC] > 0.0
    assert scores["preprocessing_seconds"] > 0.0
    assert scores["reference_queries_per_second"] > 0.0
    assert scores["query_count"] == 64.0 * 63.0
    assert scores["validation_all_pairs"] == 1.0
    assert scores["mutable_lines"] > 0.0
    assert scores["call_diversity"] > 0.0


def test_committed_shorter_distance_mutant_fails_with_named_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_shorter")
    mutant = PACK / "fixtures" / "mutants" / "shorter_distance"

    with pytest.raises(EvalError, match="shorter-than-possible distance"):
        evaluator.evaluate(mutant, stage=0)


def test_committed_valid_distance_broken_path_mutant_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_broken_path")
    mutant = PACK / "fixtures" / "mutants" / "broken_path"

    with pytest.raises(EvalError, match="path uses missing directed edge"):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_varying_container_cannot_fool_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_dir = PACK / "fixtures" / "mutants" / "hostile_container"
    hostile_module = _load_module(
        hostile_dir / "router.py",
        "test_shortest_path_hostile_container",
    )
    router = hostile_module.build_router(3, ((0, 1, 1), (1, 2, 1)))
    answer = router.query(0, 2)
    assert tuple(answer) != tuple(answer)

    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_hostile_gate")
    with pytest.raises(EvalError, match="path uses missing directed edge"):
        evaluator.evaluate(hostile_dir, stage=0)


def test_campaign_and_dynamic_bounds_parse_with_repository_loaders() -> None:
    campaign = load_campaign(CAMPAIGN)
    bounds = load_bounds(CAMPAIGN)

    assert campaign.name == "shortest-path"
    assert campaign.evaluator_path == PACK.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "small-validation",
        "large-frontier",
    ]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 2
    assert all("not a published bound" in bound.who_and_year for bound in bounds)
    assert all(bound.checked_on == "2026-08-06" for bound in bounds)


def test_contract_markers_descriptors_integer_normalization_and_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_contract")

    assert evaluator.GATE == "exact_shortest_paths"
    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "mutable_lines",
        "call_diversity",
    }
    assert evaluator._exact_int(np.int64(7), "probe") == 7
    with pytest.raises(EvalError, match="must be an integer, got bool"):
        evaluator._exact_int(True, "probe")
    assert evaluator.ceiling() is None

    baseline_source = (PACK / "baseline" / "router.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (CAMPAIGN / "spec.md").read_text(encoding="utf-8")
    assert evaluator.GATE in spec_text
    assert evaluator.METRIC in spec_text
    assert "operator.index()" in spec_text


def test_candidate_module_cannot_declare_reported_metric_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_shortest_path_self_report")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = (PACK / "baseline" / "router.py").read_text(encoding="utf-8")
    source += "\nqueries_per_second = 1e300\n"
    (candidate / "router.py").write_text(source, encoding="utf-8")

    with pytest.raises(EvalError, match="self-reported metric names: queries_per_second"):
        evaluator.evaluate(candidate, stage=0)
