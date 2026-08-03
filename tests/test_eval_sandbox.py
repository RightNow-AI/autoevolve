import time
from pathlib import Path

import pytest

from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.sandbox import run_stage

FIXTURES = Path(__file__).parent / "fixtures"
TOY_EVALUATOR = FIXTURES / "eval_toy"
CANDIDATES = FIXTURES / "eval_toy_candidates"
SMOKE = StageSpec(name="smoke", timeout_s=20.0)


def test_good_candidate_returns_metrics() -> None:
    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "good", 0, SMOKE)

    assert metrics["correct"] == 1.0
    assert metrics["score"] > 0.0


def test_broken_candidate_surfaces_failing_case() -> None:
    with pytest.raises(EvalError, match="case 0 failed"):
        run_stage(TOY_EVALUATOR, CANDIDATES / "broken", 0, SMOKE)


def test_slow_candidate_is_killed_at_wall_clock_timeout() -> None:
    started = time.monotonic()

    with pytest.raises(EvalError, match=r"timeout after 2\.0s at stage smoke"):
        run_stage(
            TOY_EVALUATOR,
            CANDIDATES / "slow",
            0,
            StageSpec(name="smoke", timeout_s=2.0),
        )

    assert time.monotonic() - started < 15.0


def test_noisy_candidate_does_not_corrupt_protocol() -> None:
    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "noisy", 0, SMOKE)

    assert metrics["correct"] == 1.0


def test_network_call_is_blocked() -> None:
    with pytest.raises(EvalError, match="network disabled in sandbox"):
        run_stage(TOY_EVALUATOR, CANDIDATES / "netcall", 0, SMOKE)


def test_environment_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_CANARY", "must-not-leak")

    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "envspy", 0, SMOKE)

    assert metrics["correct"] == 1.0
