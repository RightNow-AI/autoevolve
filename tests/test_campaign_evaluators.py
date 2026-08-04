from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "campaigns" / "arch-search" / "evaluators" / "tiny-mlp"
ALGORITHM = ROOT / "campaigns" / "algorithm-frontier" / "evaluators" / "binpack"
EQUATION = (
    ROOT
    / "campaigns"
    / "equation-discovery"
    / "evaluators"
    / "symreg-nguyen5"
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arch_baseline_passes_and_nonfinite_mutant_fails() -> None:
    evaluator = _load(ARCH / "evaluate.py", "test_campaign_arch")
    scores = evaluator.evaluate(ARCH / "baseline")

    assert scores["trained"] == 1.0
    assert scores["val_loss"] > 0.0
    assert scores["params"] > 0.0
    with pytest.raises(EvalError, match="non-finite"):
        evaluator.evaluate(ARCH / "fixtures" / "mutants" / "nonfinite")


@pytest.mark.parametrize("cell", ["uniform", "clustered"])
def test_binpack_baseline_passes_and_duplicate_mutant_fails(
    cell: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    evaluator = _load(ALGORITHM / "evaluate.py", f"test_campaign_binpack_{cell}")
    scores = evaluator.evaluate(ALGORITHM / "baseline")

    # The two program-structure descriptors ride alongside the score so the
    # archive keeps distinct heuristic shapes instead of collapsing to one cell.
    assert scores == {
        "valid": 1.0,
        "bins_used": scores["bins_used"],
        "mutable_lines": scores["mutable_lines"],
        "call_diversity": scores["call_diversity"],
    }
    assert scores["bins_used"] > 0.0
    assert scores["mutable_lines"] > 0.0
    with pytest.raises(EvalError, match="exactly once"):
        evaluator.evaluate(ALGORITHM / "fixtures" / "mutants" / "duplicate")


def test_nguyen5_baseline_passes_nonfinite_fails_and_exact_model_recovers() -> None:
    evaluator = _load(EQUATION / "evaluate.py", "test_campaign_nguyen5")
    baseline = evaluator.evaluate(EQUATION / "baseline")
    exact = evaluator.evaluate(EQUATION / "fixtures" / "mutants" / "true_nguyen5")

    assert baseline["finite"] == 1.0
    assert exact["r2_heldout"] > 0.99
    with pytest.raises(EvalError, match="non-finite"):
        evaluator.evaluate(EQUATION / "fixtures" / "mutants" / "nonfinite")


def test_pack_local_fixtures_regenerate_byte_identically(tmp_path: Path) -> None:
    arch_generator = _load(
        ARCH / "fixtures" / "make_fixtures.py",
        "test_campaign_arch_generator",
    )
    algorithm_generator = _load(
        ALGORITHM / "fixtures" / "make_fixtures.py",
        "test_campaign_algorithm_generator",
    )
    equation_generator = _load(
        EQUATION / "fixtures" / "make_fixtures.py",
        "test_campaign_equation_generator",
    )

    arch_out = tmp_path / "arch"
    algorithm_out = tmp_path / "algorithm"
    equation_out = tmp_path / "equation"
    arch_generator.write_fixtures(arch_out)
    algorithm_generator.write_fixtures(algorithm_out)
    equation_generator.write_fixtures(equation_out)

    assert (arch_out / "data.json").read_bytes() == (
        ARCH / "fixtures" / "data.json"
    ).read_bytes()
    for name in ("uniform.json", "clustered.json"):
        assert (algorithm_out / name).read_bytes() == (
            ALGORITHM / "fixtures" / name
        ).read_bytes()
    for name in ("train.json", "heldout.json"):
        assert (equation_out / name).read_bytes() == (
            EQUATION / "fixtures" / name
        ).read_bytes()

