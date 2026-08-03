"""End-to-end unit tests for the core Engine facade with injected eval seams."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoevolve.core import archive
from autoevolve.core.db import connection, transaction, utc_now
from autoevolve.core.engine import Engine
from autoevolve.core.events import load_events
from autoevolve.core.types import Budget, Contract, EvalOutcome, StageSpec


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


class StubEvaluator:
    """Minimal loader result carrying contract metadata for U1 tests."""

    def __init__(
        self,
        contract: Contract,
        baseline_files: dict[str, str] | None = None,
        ceiling: dict | None = None,
    ):
        self.contract = contract
        self.baseline_files = baseline_files or {"candidate.py": "value = 1\n"}
        self.stages = [StageSpec(name="quick", timeout_s=1.0)]
        self.gate = contract.gate
        self._ceiling = ceiling

    def ceiling(self) -> dict | None:
        return self._ceiling


class ScriptedCascade:
    """Returns outcomes in order and records directory-only calls."""

    def __init__(self, outcomes: list[EvalOutcome]):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, evaluator_dir: Path, candidate_dir: Path) -> EvalOutcome:
        assert isinstance(evaluator_dir, Path)
        assert candidate_dir.is_dir()
        self.calls.append((evaluator_dir, candidate_dir))
        if not self.outcomes:
            raise AssertionError("unexpected cascade call")
        return self.outcomes.pop(0)


def _contract(
    *,
    target: float | None = None,
    plateau_n: int = 150,
    maximize: bool = True,
) -> Contract:
    return Contract(
        goal="improve score",
        domain="unit-test",
        metric="score",
        maximize=maximize,
        baseline=None,
        target=target,
        gate="correct",
        budget=Budget(max_evals=100),
        plateau_n=plateau_n,
    )


def _outcome(
    score: float,
    *,
    gate_passed: bool = True,
    error: str | None = None,
) -> EvalOutcome:
    return EvalOutcome(
        gate_passed=gate_passed,
        scores={"correct": float(gate_passed), "score": score},
        stage_reached=0,
        error=error,
    )


def _make_engine(
    home: Path,
    contract: Contract,
    outcomes: list[EvalOutcome],
    *,
    baseline_files: dict[str, str] | None = None,
    ceiling: dict | None = None,
) -> tuple[Engine, ScriptedCascade, Path]:
    evaluator = StubEvaluator(contract, baseline_files, ceiling)
    cascade = ScriptedCascade(outcomes)
    evaluator_dir = home / "evaluator"
    engine = Engine(home, cascade=cascade, evaluator_loader=lambda _: evaluator)
    return engine, cascade, evaluator_dir


def test_open_run_records_three_run_median_and_seed_program(home: Path) -> None:
    engine, cascade, evaluator_dir = _make_engine(
        home,
        _contract(),
        [_outcome(1.0), _outcome(3.0), _outcome(2.0)],
    )
    opened = engine.open_run(
        "improve score",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=8),
        seed=123,
    )
    assert opened["contract"].baseline == 2.0
    assert len(cascade.calls) == 3
    with connection(home) as conn:
        seed = conn.execute(
            "SELECT id, operator, parent_id FROM programs WHERE run_id = ?",
            (opened["run_id"],),
        ).fetchone()
        score = conn.execute(
            "SELECT value, stage FROM scores WHERE program_id = ? AND metric = 'score'",
            (seed["id"],),
        ).fetchone()
    assert seed["operator"] == "seed" and seed["parent_id"] is None
    assert float(score["value"]) == 2.0 and int(score["stage"]) == 0
    assert engine.best(opened["run_id"])[0]["program_id"] == seed["id"]


def test_open_run_above_ceiling_is_infeasible(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(target=10.0),
        [_outcome(1.0), _outcome(1.0), _outcome(1.0)],
        ceiling={"metric": "score", "value": 5.0, "method": "proof"},
    )
    opened = engine.open_run(
        "impossible target",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=10),
    )
    assert opened["status"] == "infeasible"
    assert opened["feasibility"]["value"] == 5.0
    assert engine.run_status(opened["run_id"])["status"] == "infeasible"


def test_open_run_refuses_unbounded_budget_before_loading_evaluator(home: Path) -> None:
    def unexpected_loader(_: Path) -> object:
        raise AssertionError("loader should not run")

    engine = Engine(home, cascade=lambda *_: _outcome(1.0), evaluator_loader=unexpected_loader)
    with pytest.raises(ValueError, match="at least one bound"):
        engine.open_run(
            "unbounded",
            evaluator_ref=home / "evaluator",
            budget=Budget(max_evals=None),
        )


def test_submit_child_gate_pass_updates_archive_and_bandit(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(),
        [_outcome(1.0), _outcome(1.0), _outcome(1.0), _outcome(2.0)],
    )
    run_id = engine.open_run(
        "improve",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=5),
        seed=11,
    )["run_id"]
    bundle = engine.next_parent(run_id, 0)
    result = engine.submit_child(
        run_id,
        bundle.parent.id,
        "diff",
        {"candidate.py": "value = 2\n"},
        "measured improvement",
    )
    assert {
        "program_id",
        "gate_passed",
        "scores",
        "fitness",
        "archive_improved",
        "best_fitness",
        "plateau",
        "budget_remaining",
    } <= result.keys()
    assert result["gate_passed"]
    assert result["scores"]["score"] == 2.0
    assert result["fitness"] == 2.0
    assert result["archive_improved"]
    assert engine.best(run_id)[0]["program_id"] == result["program_id"]
    assert [row["program_id"] for row in engine.lineage(result["program_id"])] == [
        bundle.parent.id,
        result["program_id"],
    ]
    with connection(home) as conn:
        arm = conn.execute(
            "SELECT pulls, improvements, mean_gain FROM operators "
            "WHERE domain = 'unit-test' AND name = 'diff'"
        ).fetchone()
    assert int(arm["pulls"]) == 1
    assert int(arm["improvements"]) == 1
    assert float(arm["mean_gain"]) == pytest.approx(1.0)


def test_gate_fail_records_zero_and_touches_neither_archive_nor_bandit(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(),
        [
            _outcome(1.0),
            _outcome(1.0),
            _outcome(1.0),
            _outcome(99.0, gate_passed=False, error="parity mismatch"),
        ],
    )
    run_id = engine.open_run(
        "improve",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=5),
        seed=12,
    )["run_id"]
    bundle = engine.next_parent(run_id, 0)
    result = engine.submit_child(
        run_id,
        bundle.parent.id,
        "diff",
        {"candidate.py": "wrong = True\n"},
    )
    assert not result["gate_passed"]
    assert result["scores"]["score"] == 0.0
    assert not result["archive_improved"]
    with connection(home) as conn:
        assert len(archive.current_elites(conn, run_id)) == 1
        arms = conn.execute(
            "SELECT pulls, improvements FROM operators WHERE domain = 'unit-test'"
        ).fetchall()
        events = load_events(conn, run_id, "gate_failed")
    assert all(int(row["pulls"]) == 0 and int(row["improvements"]) == 0 for row in arms)
    assert events[-1]["payload"]["reason"] == "parity mismatch"


def test_evolve_block_violation_rejects_without_program(home: Path) -> None:
    baseline = {
        "candidate.py": (
            "frozen = 1\n"
            "# EVOLVE-BLOCK-START\n"
            "value = 1\n"
            "# EVOLVE-BLOCK-END\n"
        )
    }
    engine, cascade, evaluator_dir = _make_engine(
        home,
        _contract(),
        [_outcome(1.0), _outcome(1.0), _outcome(1.0)],
        baseline_files=baseline,
    )
    run_id = engine.open_run(
        "improve",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=5),
    )["run_id"]
    bundle = engine.next_parent(run_id, 0)
    result = engine.submit_child(
        run_id,
        bundle.parent.id,
        "diff",
        {
            "candidate.py": (
                "frozen = 2\n"
                "# EVOLVE-BLOCK-START\n"
                "value = 2\n"
                "# EVOLVE-BLOCK-END\n"
            )
        },
    )
    assert result["rejected"]
    assert "frozen text changed" in result["reason"]
    assert len(cascade.calls) == 3
    with connection(home) as conn:
        programs = conn.execute(
            "SELECT COUNT(*) AS count FROM programs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        event = load_events(conn, run_id, "gate_failed")[-1]
    assert int(programs["count"]) == 1
    assert event["payload"]["reason"] == "evolve_block_violation"


def test_budget_exhaustion_flips_status(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(),
        [_outcome(1.0), _outcome(1.0), _outcome(1.0), _outcome(2.0)],
    )
    run_id = engine.open_run(
        "one try",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=1),
    )["run_id"]
    bundle = engine.next_parent(run_id, 0)
    result = engine.submit_child(
        run_id,
        bundle.parent.id,
        "diff",
        {"candidate.py": "value = 2\n"},
    )
    assert result["status"] == "budget_exhausted"
    assert result["budget_remaining"]["max_evals"] == 0
    assert engine.run_status(run_id)["status"] == "budget_exhausted"


def test_plateau_flips_after_configured_non_improvements(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(plateau_n=2),
        [
            _outcome(1.0),
            _outcome(1.0),
            _outcome(1.0),
            _outcome(1.0),
            _outcome(1.0),
        ],
    )
    run_id = engine.open_run(
        "plateau quickly",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=10),
    )["run_id"]
    first = engine.next_parent(run_id, 0)
    first_result = engine.submit_child(
        run_id,
        first.parent.id,
        "diff",
        {"candidate.py": "attempt = 1\n"},
    )
    assert first_result["status"] == "open"
    second = engine.next_parent(run_id, 0)
    second_result = engine.submit_child(
        run_id,
        second.parent.id,
        "diff",
        {"candidate.py": "attempt = 2\n"},
    )
    assert second_result["status"] == "plateau"
    assert second_result["plateau"]
    assert engine.run_status(run_id)["plateau"]["non_improving"] == 2


def test_run_status_paths_and_event_sequences_are_stable(home: Path) -> None:
    engine, _, evaluator_dir = _make_engine(
        home,
        _contract(),
        [_outcome(1.0), _outcome(1.0), _outcome(1.0)],
    )
    run_id = engine.open_run(
        "inspect status",
        evaluator_ref=evaluator_dir,
        budget=Budget(max_evals=3),
        workers=2,
    )["run_id"]
    status = engine.run_status(run_id)
    assert status["curve"] == [[0, 1.0]]
    assert len(status["islands"]) == 2
    gif_path = Path(status["artifacts"]["gif"])
    assert gif_path.name == "evolution.gif"
    assert gif_path.parent.name == run_id
    with connection(home) as conn:
        events = load_events(conn, run_id)
    assert [event["seq"] for event in events] == list(range(len(events)))


def test_discoveries_rank_keyword_overlap_then_recency(home: Path) -> None:
    engine = Engine(home)
    with transaction(home) as conn:
        conn.executemany(
            "INSERT INTO discoveries(id, domain, text, source_run, source_programs, "
            "created_at) VALUES (?, 'kernels', ?, NULL, ?, ?)",
            [
                ("d0000000001", "old unrelated note", "[]", "2026-01-01T00:00:00+00:00"),
                (
                    "d0000000002",
                    "tile kernels for speed",
                    json.dumps(["p1"]),
                    "2026-01-02T00:00:00+00:00",
                ),
                (
                    "d0000000003",
                    "new unrelated note",
                    "[]",
                    utc_now(),
                ),
            ],
        )
    rows = engine.discoveries("kernels", "speed kernels")
    assert rows[0]["id"] == "d0000000002"
    assert rows[0]["source_programs"] == ["p1"]
