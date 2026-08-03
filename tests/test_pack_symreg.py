from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "symbolic-regression"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_has_real_headroom() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_symreg_baseline")
    scores = evaluator.evaluate(PACK / "baseline", stage=1)
    assert scores["finite"] == 1.0
    assert scores["r2_heldout"] < 0.9


def test_true_nguyen7_mutant_reaches_the_ladder_top() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_symreg_true")
    candidate = PACK / "fixtures" / "mutants" / "true_nguyen7"
    scores = evaluator.evaluate(candidate, stage=1)
    assert scores["finite"] == 1.0
    assert scores["r2_heldout"] > 0.99


def test_complexity_penalty_reduces_fitness_for_equal_predictions() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_symreg_complexity")
    baseline = evaluator.evaluate(PACK / "baseline", stage=1)
    candidate_dir = PACK / "fixtures" / "mutants" / "equivalent_but_complex"
    complex_candidate = evaluator.evaluate(candidate_dir, stage=1)
    assert complex_candidate["r2_heldout"] == pytest.approx(baseline["r2_heldout"])
    assert complex_candidate["complexity"] > baseline["complexity"]
    assert complex_candidate["fitness"] < baseline["fitness"]


def test_heldout_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_symreg_fixture_generator",
    )
    generator.write_fixtures(tmp_path)
    committed = (PACK / "fixtures" / "heldout.json").read_bytes()
    assert (tmp_path / "heldout.json").read_bytes() == committed

