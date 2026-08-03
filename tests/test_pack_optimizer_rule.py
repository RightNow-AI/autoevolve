from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "optimizer-rule"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_passes_every_stage_and_stage_zero_has_metric() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_optimizer_rule_baseline")

    for stage in range(len(evaluator.STAGES)):
        scores = evaluator.evaluate(PACK / "baseline", stage=stage)
        assert scores[evaluator.GATE] == 1.0
        assert scores["steps"] == 300.0
        assert scores["train_loss"] > 0.0
        assert 0.0 <= scores["val_accuracy"] <= 1.0

    stage_zero = evaluator.evaluate(PACK / "baseline", stage=0)
    assert evaluator.METRIC in stage_zero
    assert evaluator.METRIC == "val_loss"
    assert evaluator.MAXIMIZE is False


def test_baseline_is_exactly_deterministic() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_optimizer_rule_determinism")

    first = evaluator.evaluate(PACK / "baseline", stage=1)
    second = evaluator.evaluate(PACK / "baseline", stage=1)

    assert first == second


def test_diverging_mutant_fails_with_step_and_parameter() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_optimizer_rule_diverging")
    mutant = PACK / "fixtures" / "mutants" / "diverging"

    with pytest.raises(
        EvalError,
        match=r"step \d+ parameter \w+.*non-finite",
    ):
        evaluator.evaluate(mutant, stage=0)


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_optimizer_rule_fixture_generator",
    )

    generator.write_fixtures(tmp_path)

    assert (tmp_path / "data.json").read_bytes() == (
        PACK / "fixtures" / "data.json"
    ).read_bytes()


def test_pack_contract_markers_and_spec() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_optimizer_rule_contract")

    assert evaluator.STAGES
    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)
    assert len(evaluator._stage_seeds(0)) == 1
    assert len(evaluator._stage_seeds(1)) == 3
    assert evaluator.ceiling() is None

    baseline_source = (PACK / "baseline" / "rule.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (PACK / "spec.md").read_text(encoding="utf-8")
    assert evaluator.GATE in spec_text
    assert evaluator.METRIC in spec_text


def test_hostile_list_subclass_cannot_change_verdict(tmp_path: Path) -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_optimizer_rule_hostile")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "rule.py").write_text(
        """from __future__ import annotations

import numpy as np


class Hostile(list):
    def __init__(self, values):
        super().__init__(values)
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        if self.reads == 1:
            return super().__getitem__(index)
        return np.full_like(super().__getitem__(0), np.nan)


def init_state(shape: tuple[int, ...]) -> dict:
    del shape
    return {}


def update(param, grad, state, step):
    del state, step
    return Hostile([param - 0.05 * grad, {}])
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="update must return an exact tuple"):
        evaluator.evaluate(candidate, stage=0)
