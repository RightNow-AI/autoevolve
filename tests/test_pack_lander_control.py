from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "lander-control"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_passes_every_stage_and_reports_primary_metric() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lander_baseline")
    expected_scenarios = (3.0, 6.0)
    for stage, scenario_count in enumerate(expected_scenarios):
        scores = evaluator.evaluate(PACK / "baseline", stage=stage)
        assert scores[evaluator.GATE] == 1.0
        assert evaluator.METRIC in scores
        assert 0.0 < scores[evaluator.METRIC] < 1.0
        assert scores["mean_touchdown_speed"] <= 2.0
        assert scores["scenarios_landed"] == scenario_count


def test_crash_mutant_names_touchdown_speed_failure() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lander_crash")
    mutant = PACK / "fixtures" / "mutants" / "crash"
    with pytest.raises(EvalError, match=r"scenario-01.*vertical speed.*\|vy\|="):
        evaluator.evaluate(mutant, stage=0)


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_lander_fixture_generator",
    )
    generator.write_fixtures(tmp_path)
    regenerated = (tmp_path / "scenarios.json").read_bytes()
    assert regenerated == (PACK / "fixtures" / "scenarios.json").read_bytes()


def test_contract_shape_markers_and_spec_names() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lander_contract")
    assert evaluator.STAGES
    assert evaluator.GATE == "landed"
    assert evaluator.METRIC == "fuel_efficiency"
    assert evaluator.MAXIMIZE is True
    assert evaluator.ceiling() is None
    assert all(isinstance(stage, StageSpec) for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)

    baseline_source = (PACK / "baseline" / "policy.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (PACK / "spec.md").read_text(encoding="utf-8")
    assert evaluator.GATE in spec_text
    assert evaluator.METRIC in spec_text


def test_hostile_list_subclass_is_snapshotted_before_control_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lander_hostile")
    baseline = _load_module(PACK / "baseline" / "policy.py", "test_lander_policy")

    class HostileControl(list[float]):
        def __init__(self, throttle: float, gimbal: float) -> None:
            super().__init__((throttle, gimbal))
            self.reads = [0, 0]

        def __getitem__(self, index: int) -> float:
            self.reads[index] += 1
            if self.reads[index] > 1:
                return math.nan
            return super().__getitem__(index)

    returned_controls: list[HostileControl] = []

    def hostile_act(state: dict[str, float], time_s: float) -> HostileControl:
        throttle, gimbal = baseline.act(state, time_s)
        control = HostileControl(throttle, gimbal)
        returned_controls.append(control)
        return control

    probe = HostileControl(0.5, 0.0)
    assert probe[0] == 0.5
    assert math.isnan(probe[0])

    monkeypatch.setattr(evaluator, "_load_policy", lambda _candidate_dir: hostile_act)
    scores = evaluator.evaluate(PACK / "baseline", stage=0)
    assert scores[evaluator.GATE] == 1.0
    assert returned_controls
    assert all(control.reads == [0, 0] for control in returned_controls)


def test_non_finite_control_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lander_non_finite")
    assert evaluator._normalize_control((1e300, -1e300), "probe", 0.0) == (1.0, -1.0)

    def non_finite_act(_state: dict[str, float], _time_s: float) -> tuple[float, float]:
        return math.nan, 0.0

    monkeypatch.setattr(evaluator, "_load_policy", lambda _candidate_dir: non_finite_act)
    with pytest.raises(EvalError, match=r"scenario-01.*non-finite throttle"):
        evaluator.evaluate(PACK / "baseline", stage=0)
