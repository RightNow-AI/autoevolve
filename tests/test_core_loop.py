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


def test_loop_skips_cycles_on_skip_cycle_errors(home: Path) -> None:
    """An operator failure carrying skip_cycle must not kill the loop."""

    class SkipError(Exception):
        skip_cycle = True

    class FlakyOperator:
        name = "diff"

        def __init__(self):
            self.calls = 0

        def propose(self, bundle, ctx):
            self.calls += 1
            if self.calls == 1:
                raise SkipError("bad model response")
            return Proposal({"candidate.py": "value = 3\n"}, "recovered")

    engine, _cascade, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "skip once then recover",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=1),
        seed=23,
    )["run_id"]
    operator = FlakyOperator()

    summary = run_worker_loop(engine, run_id, lambda name: operator, max_cycles=4)

    assert summary["skips"] == 1
    assert summary["submissions"] == 1
    assert operator.calls == 2


def test_loop_raises_after_consecutive_skip_cap(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoevolve.core import loop as loop_module

    class SkipError(Exception):
        skip_cycle = True

    class AlwaysSkip:
        name = "diff"

        def propose(self, bundle, ctx):
            raise SkipError("never works")

    monkeypatch.setattr(loop_module, "_MAX_CONSECUTIVE_SKIPS", 3)
    engine, _cascade, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "always skipping",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=5),
        seed=29,
    )["run_id"]

    with pytest.raises(RuntimeError, match="consecutive skipped cycles"):
        run_worker_loop(engine, run_id, lambda name: AlwaysSkip())


def test_elite_less_island_progresses_without_impossible_crossover(home: Path) -> None:
    class SkipError(Exception):
        skip_cycle = True

    class PartnerAwareOperator:
        def __init__(self, name: str) -> None:
            self.name = name

        def propose(self, bundle: object, ctx: SimpleNamespace) -> Proposal:
            crossover_parent = getattr(bundle, "crossover_parent", None)
            if self.name == "crossover" and crossover_parent is None:
                raise SkipError("crossover requires a partner")
            return Proposal({"candidate.py": "value = 3\n"}, "made progress")

    engine, _cascade, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "joined worker crossover fallback",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=2),
        workers=2,
        seed=43,
    )["run_id"]
    assert engine.join_run(run_id, "worker-zero")["island"] == 0
    joined_island = engine.join_run(run_id, "worker-one")["island"]
    assert joined_island == 1

    first = engine.next_parent(run_id, 0)
    engine.submit_child(
        run_id,
        first.parent.id,
        "agentic",
        {"candidate.py": "value = 2\n"},
    )
    requested: list[str] = []

    def get_operator(name: str) -> PartnerAwareOperator:
        requested.append(name)
        return PartnerAwareOperator(name)

    summary = run_worker_loop(
        engine,
        run_id,
        get_operator,
        max_cycles=30,
        island=joined_island,
    )

    assert requested == ["diff"]
    assert summary["skips"] == 0
    assert summary["submissions"] == 1
    assert summary["status"] == "budget_exhausted"
