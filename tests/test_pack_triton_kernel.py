from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "triton-kernel"


def _load_evaluator(name: str) -> ModuleType:
    path = PACK / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "triton"
        or name.startswith("triton.")
    }


def test_mock_baseline_passes_numpy_parity_without_gpu_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    monkeypatch.delenv("AUTOEVOLVE_CELL", raising=False)
    evaluator = _load_evaluator("test_triton_mock_baseline")
    before = _gpu_modules()
    scores = evaluator.evaluate(PACK / "baseline")
    assert scores["mock_parity"] == 1.0
    assert scores["mock_score"] > 0.0
    assert set(scores) == {"mock_parity", "mock_score"}
    assert _gpu_modules() == before


def test_wrong_scale_mutant_fails_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    monkeypatch.delenv("AUTOEVOLVE_CELL", raising=False)
    evaluator = _load_evaluator("test_triton_wrong_scale")
    mutant = PACK / "fixtures" / "mutants" / "wrong_scale"
    with pytest.raises(EvalError, match="vector-1024"):
        evaluator.evaluate(mutant)


def test_contract_constants_are_well_formed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    monkeypatch.delenv("AUTOEVOLVE_CELL", raising=False)
    evaluator = _load_evaluator("test_triton_contract")
    assert evaluator.GATE == "mock_parity"
    assert evaluator.STAGES
    assert all(isinstance(stage, StageSpec) for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)
    assert evaluator.ceiling() is None


@pytest.mark.parametrize(
    ("cell", "size", "alpha", "operation"),
    [
        ("add-1k", 1_024, 1.0, "add"),
        ("add-8k", 8_192, 1.0, "add"),
        ("scale-1k", 1_024, 0.375, "scale"),
        ("scale-8k", 8_192, -1.25, "scale"),
    ],
)
def test_cell_selects_only_its_size_and_operation(
    cell: str,
    size: int,
    alpha: float,
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    evaluator = _load_evaluator(f"test_triton_cell_{cell}")

    cases = evaluator._load_cases(evaluator._selected_cell())
    assert [(name, x.size, case_alpha) for name, x, _, case_alpha in cases] == [
        (f"vector-{size}-{operation}", size, alpha)
    ]
    scores = evaluator.evaluate(PACK / "baseline")
    assert scores["mock_parity"] == 1.0
    assert scores["mock_score"] > 0.0


def test_unknown_cell_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    monkeypatch.setenv("AUTOEVOLVE_CELL", "unknown")
    evaluator = _load_evaluator("test_triton_unknown_cell")

    with pytest.raises(EvalError, match="AUTOEVOLVE_CELL must be one of"):
        evaluator.evaluate(PACK / "baseline")


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    path = PACK / "fixtures" / "make_fixtures.py"
    spec = importlib.util.spec_from_file_location("test_triton_fixture_generator", path)
    assert spec is not None
    assert spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.write_fixtures(tmp_path)
    committed = (PACK / "fixtures" / "cases.json").read_bytes()
    assert (tmp_path / "cases.json").read_bytes() == committed
