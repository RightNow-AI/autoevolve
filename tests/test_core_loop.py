"""Worker-loop tests with duck-typed mutation operators."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoevolve.core.engine import Engine
from autoevolve.core.loop import run_worker_loop
from autoevolve.core.types import Budget, Contract, EvalOutcome, Proposal, StageSpec


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


class Evaluator:
    def __init__(self):
        self.contract = Contract(
            goal="loop",
            domain="loop-test",
            metric="score",
            maximize=True,
            baseline=None,
            target=None,
            gate="correct",
            budget=Budget(max_evals=10),
        )
        self.baseline_files = {"candidate.py": "value = 1\n"}
        self.stages = [StageSpec("quick", 1.0)]
        self.gate = "correct"

    @staticmethod
    def ceiling() -> None:
        return None


class Cascade:
    def __init__(self):
        self.calls = 0

    def __call__(self, evaluator_dir: Path, candidate_dir: Path) -> EvalOutcome:
        assert evaluator_dir.name == "evaluator"
        assert (candidate_dir / "candidate.py").is_file()
        self.calls += 1
        score = 1.0 if self.calls <= 3 else 2.0
        return EvalOutcome(True, {"correct": 1.0, "score": score}, 0)


class CapturingOperator:
    def __init__(self):
        self.calls: list[tuple[object, SimpleNamespace]] = []

    def propose(self, bundle: object, ctx: SimpleNamespace) -> Proposal:
        self.calls.append((bundle, ctx))
        return Proposal({"candidate.py": "value = 2\n"}, "loop proposal")


def _engine(home: Path) -> tuple[Engine, Cascade, Path]:
    evaluator = Evaluator()
    cascade = Cascade()
    path = home / "evaluator"
    engine = Engine(home, cascade=cascade, evaluator_loader=lambda _: evaluator)
    return engine, cascade, path


def test_worker_loop_proposes_submits_and_returns_artifacts(home: Path) -> None:
    engine, cascade, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "loop once",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=1),
        seed=17,
    )["run_id"]
    operator = CapturingOperator()
    requested: list[str] = []

    def get_operator(name: str) -> CapturingOperator:
        requested.append(name)
        return operator

    summary = run_worker_loop(engine, run_id, get_operator, max_cycles=5)
    assert requested == ["agentic"]
    assert summary["cycles"] == 1
    assert summary["submissions"] == 1
    assert summary["status"] == "budget_exhausted"
    assert summary["last_result"]["archive_improved"]
    assert set(summary["artifacts"]) == {"gif", "poster", "dashboard"}
    assert cascade.calls == 4
    _, context = operator.calls[0]
    assert context.contract.metric == "score"
    assert context.run_id == run_id
    assert context.cycle == 0
    assert context.workdir == home.resolve()
    assert context.rng.random() == engine.decision_rng(
        run_id,
        "operator:agentic",
        5,
    ).random()


def test_worker_loop_honors_zero_cycle_cap(home: Path) -> None:
    engine, cascade, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "do not cycle",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=3),
    )["run_id"]

    def unexpected(_: str) -> object:
        raise AssertionError("operator should not be loaded")

    summary = run_worker_loop(engine, run_id, unexpected, max_cycles=0)
    assert summary["cycles"] == 0
    assert summary["submissions"] == 0
    assert summary["status"] == "open"
    assert cascade.calls == 3


def test_worker_loop_rejects_negative_cycle_cap(home: Path) -> None:
    engine, _, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "invalid cap",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=3),
    )["run_id"]
    with pytest.raises(ValueError, match="non-negative"):
        run_worker_loop(engine, run_id, lambda _: object(), max_cycles=-1)
