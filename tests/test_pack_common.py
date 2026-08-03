from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import StageSpec

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = ROOT / "evaluators"
PACKS = (
    ("lossless-compression", "codec.py", "compression_ratio"),
    ("python-speedup", "pipeline.py", "speedup"),
    ("triton-kernel", "kernel.py", "tflops"),
    ("routing-heuristic", "heuristic.py", "tour_cost"),
    ("symbolic-regression", "model.py", "fitness"),
)


def _load_evaluator(pack_name: str) -> ModuleType:
    path = EVALUATORS / pack_name / "evaluate.py"
    module_name = f"test_pack_common_{pack_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("pack_name", "entry_name", "headline_metric"), PACKS)
def test_pack_contract_and_baseline_stage_zero(
    pack_name: str,
    entry_name: str,
    headline_metric: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    pack_dir = EVALUATORS / pack_name
    evaluator = _load_evaluator(pack_name)

    assert evaluator.STAGES
    assert all(isinstance(stage, StageSpec) for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)

    scores = evaluator.evaluate(pack_dir / "baseline", stage=0)
    assert scores
    assert evaluator.GATE in scores
    assert scores[evaluator.GATE] == 1.0

    baseline_source = (pack_dir / "baseline" / entry_name).read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (pack_dir / "spec.md").read_text(encoding="utf-8")
    assert spec_text.strip()
    assert evaluator.GATE in spec_text
    assert headline_metric in spec_text


def test_every_pack_declares_primary_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pack without METRIC and MAXIMIZE lets the engine guess the wrong hill."""

    monkeypatch.setenv("AUTOEVOLVE_FORCE_TRITON_MOCK", "1")
    for pack_name, _entry, _headline in PACKS:
        module = _load_evaluator(pack_name)
        metric = getattr(module, "METRIC", None)
        maximize = getattr(module, "MAXIMIZE", None)
        assert isinstance(metric, str) and metric, f"{pack_name} must declare METRIC"
        assert isinstance(maximize, bool), f"{pack_name} must declare MAXIMIZE"
        scores = module.evaluate(EVALUATORS / pack_name / "baseline", stage=0)
        assert metric in scores, f"{pack_name} METRIC must be measured at stage 0"
