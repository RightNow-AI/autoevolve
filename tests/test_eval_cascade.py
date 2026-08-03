from pathlib import Path

import pytest

from autoevolve.core.types import StageSpec
from autoevolve.eval import cascade
from autoevolve.eval.contract import Evaluator


def _evaluator() -> Evaluator:
    return Evaluator(
        dir=Path("unused-evaluator"),
        stages=[
            StageSpec(name="smoke", timeout_s=1.0),
            StageSpec(name="full", timeout_s=1.0),
        ],
        gate="correct",
        has_ceiling=False,
        spec_text="",
    )


def test_two_stage_progression_uses_deepest_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    stages_seen: list[int] = []

    def fake_run_stage(
        evaluator_dir: Path,
        candidate_dir: Path,
        stage: int,
        spec: StageSpec,
    ) -> dict[str, float]:
        del evaluator_dir, candidate_dir, spec
        stages_seen.append(stage)
        return {"correct": 1.0, "score": float(stage + 10)}

    monkeypatch.setattr(cascade, "run_stage", fake_run_stage)

    outcome = cascade.run_cascade(_evaluator(), Path("unused-candidate"))

    assert stages_seen == [0, 1]
    assert outcome.gate_passed is True
    assert outcome.stage_reached == 1
    assert outcome.scores == {"correct": 1.0, "score": 11.0}
    assert outcome.error is None


def test_stage_zero_gate_failure_stops_cascade(monkeypatch: pytest.MonkeyPatch) -> None:
    stages_seen: list[int] = []

    def fake_run_stage(
        evaluator_dir: Path,
        candidate_dir: Path,
        stage: int,
        spec: StageSpec,
    ) -> dict[str, float]:
        del evaluator_dir, candidate_dir, spec
        stages_seen.append(stage)
        return {"correct": 0.0, "score": 999.0}

    monkeypatch.setattr(cascade, "run_stage", fake_run_stage)

    outcome = cascade.run_cascade(_evaluator(), Path("unused-candidate"))

    assert stages_seen == [0]
    assert outcome.gate_passed is False
    assert outcome.stage_reached == 0
    assert outcome.scores == {}
    assert outcome.error is not None and "correct" in outcome.error


def test_missing_gate_metric_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_stage(
        evaluator_dir: Path,
        candidate_dir: Path,
        stage: int,
        spec: StageSpec,
    ) -> dict[str, float]:
        del evaluator_dir, candidate_dir, stage, spec
        return {"score": 10.0}

    monkeypatch.setattr(cascade, "run_stage", fake_run_stage)

    outcome = cascade.run_cascade(_evaluator(), Path("unused-candidate"))

    assert outcome.gate_passed is False
    assert outcome.stage_reached == 0
    assert outcome.error == "gate metric missing: correct"


def test_non_numeric_metric_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_stage(
        evaluator_dir: Path,
        candidate_dir: Path,
        stage: int,
        spec: StageSpec,
    ) -> dict[str, float]:
        del evaluator_dir, candidate_dir, stage, spec
        return {"correct": 1.0, "score": "not-a-number"}  # type: ignore[dict-item]

    monkeypatch.setattr(cascade, "run_stage", fake_run_stage)

    outcome = cascade.run_cascade(_evaluator(), Path("unused-candidate"))

    assert outcome.gate_passed is False
    assert outcome.stage_reached == 0
    assert outcome.scores == {}
    assert outcome.error == "metric score is not numeric at stage 0"
