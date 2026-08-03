"""Recorded parent sampling and migration replay tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoevolve.core.db import connection
from autoevolve.core.engine import Engine
from autoevolve.core.events import load_events
from autoevolve.core.replay import replay
from autoevolve.core.types import Budget, Contract, Descriptor, EvalOutcome, StageSpec


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


class ReplayEvaluator:
    def __init__(self, descriptors: bool = False):
        configured = (
            [Descriptor("bucket", "bucket", bins=2, lo=0.0, hi=1.0)]
            if descriptors
            else []
        )
        self.contract = Contract(
            goal="replay",
            domain="replay-test",
            metric="score",
            maximize=True,
            baseline=None,
            target=None,
            gate="correct",
            budget=Budget(max_evals=100),
            descriptors=configured,
            plateau_n=100,
        )
        self.baseline_files = {"candidate.py": "bucket = 0\n"}
        self.stages = [StageSpec("quick", 1.0)]
        self.gate = "correct"

    @staticmethod
    def ceiling() -> None:
        return None


class ContentCascade:
    def __init__(self, descriptors: bool = False):
        self.descriptors = descriptors

    def __call__(self, evaluator_dir: Path, candidate_dir: Path) -> EvalOutcome:
        assert evaluator_dir.name == "evaluator"
        text = (candidate_dir / "candidate.py").read_text(encoding="utf-8")
        bucket = 1.0 if "bucket = 1" in text else 0.0
        score = 2.0 if bucket == 1.0 else 1.0
        scores = {"correct": 1.0, "score": score}
        if self.descriptors:
            scores["bucket"] = bucket
        return EvalOutcome(True, scores, 0)


def _engine(home: Path, descriptors: bool = False) -> tuple[Engine, Path]:
    evaluator = ReplayEvaluator(descriptors)
    engine = Engine(
        home,
        cascade=ContentCascade(descriptors),
        evaluator_loader=lambda _: evaluator,
    )
    return engine, home / "evaluator"


def test_replay_reproduces_recorded_parent_sampling(home: Path) -> None:
    engine, evaluator_dir = _engine(home)
    run_id = engine.open_run(
        "sample deterministically",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=5),
        seed=1234,
    )["run_id"]
    bundle = engine.next_parent(run_id, 0)
    engine.submit_child(
        run_id,
        bundle.parent.id,
        "diff",
        {"candidate.py": "value = 2\n"},
    )
    engine.next_parent(run_id, 0)
    summary = replay(home, run_id)
    assert summary["sampling_checks"] == 2
    assert summary["migration_checks"] == 0
    assert summary["event_sequence_checks"] > 0


def test_replay_reproduces_migration_after_25_island_submissions(home: Path) -> None:
    engine, evaluator_dir = _engine(home, descriptors=True)
    run_id = engine.open_run(
        "migrate deterministically",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=40),
        workers=2,
        seed=99,
    )["run_id"]

    neighbor_parent = engine.next_parent(run_id, 1)
    engine.submit_child(
        run_id,
        neighbor_parent.parent.id,
        "diff",
        {"candidate.py": "bucket = 1\n"},
    )
    for index in range(25):
        bundle = engine.next_parent(run_id, 0)
        engine.submit_child(
            run_id,
            bundle.parent.id,
            "diff",
            {"candidate.py": f"bucket = 0\nattempt = {index}\n"},
        )

    migrated = engine.next_parent(run_id, 0)
    assert migrated.parent.island == 1
    with connection(home) as conn:
        migration_events = load_events(conn, run_id, "migration")
    assert len(migration_events) == 1
    assert migration_events[0]["payload"]["submission_count"] == 25
    summary = replay(home, run_id)
    assert summary["migration_checks"] == 1
    assert summary["sampling_checks"] == 27


def test_replay_rejects_unknown_run(home: Path) -> None:
    Engine(home)
    with pytest.raises(KeyError, match="unknown run"):
        replay(home, "rmissing0000")
