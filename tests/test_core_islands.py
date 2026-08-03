"""Island assignment and deterministic migration cadence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoevolve.core import islands
from autoevolve.core.db import init_db, transaction, utc_now
from autoevolve.core.events import append_event
from autoevolve.core.types import Budget, Contract


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def _create_run(home: Path, count: int = 3) -> str:
    init_db(home)
    run_id = "r0000000001"
    contract = Contract(
        goal="goal",
        domain="domain",
        metric="score",
        maximize=True,
        baseline=1.0,
        target=None,
        gate="correct",
        budget=Budget(max_evals=100),
    )
    with transaction(home) as conn:
        conn.execute(
            "INSERT INTO runs(id, goal_text, domain, contract_json, status, budget_json, "
            "seed, evaluator_ref, created_at) VALUES (?, ?, ?, ?, 'open', ?, 5, NULL, ?)",
            (
                run_id,
                contract.goal,
                contract.domain,
                contract.to_json(),
                json.dumps({"max_evals": 100}),
                utc_now(),
            ),
        )
        islands.create_islands(conn, run_id, count)
    return run_id


def test_join_assignment_is_round_robin(home: Path) -> None:
    run_id = _create_run(home, 3)
    assignments: list[int] = []
    with transaction(home) as conn:
        for index in range(5):
            assigned = islands.assign_island(conn, run_id, f"worker-{index}")
            assignments.append(assigned)
            append_event(
                conn,
                run_id,
                "worker_joined",
                {"island": assigned, "runtime": f"worker-{index}"},
            )
    assert assignments == [0, 1, 2, 0, 1]


def test_neighbor_wraps_around_ring(home: Path) -> None:
    run_id = _create_run(home, 4)
    with transaction(home) as conn:
        assert islands.neighbor_island(conn, run_id, 0) == 1
        assert islands.neighbor_island(conn, run_id, 3) == 0


def test_migration_is_due_once_at_each_25_submission_boundary(home: Path) -> None:
    run_id = _create_run(home, 2)
    now = utc_now()
    with transaction(home) as conn:
        conn.execute(
            "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
            "cell_key, created_at) VALUES ('pseed000001', ?, NULL, 'seed', 'ref', 0, '0', ?)",
            (run_id, now),
        )
        for index in range(24):
            conn.execute(
                "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
                "cell_key, created_at) VALUES (?, ?, 'pseed000001', 'diff', 'ref', 0, "
                "'0', ?)",
                (f"pchild{index:04d}", run_id, now),
            )
        assert not islands.migration_due(conn, run_id, 0)
        conn.execute(
            "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
            "cell_key, created_at) VALUES ('pchild0024', ?, 'pseed000001', 'diff', "
            "'ref', 0, '0', ?)",
            (run_id, now),
        )
        assert islands.migration_due(conn, run_id, 0)
        append_event(
            conn,
            run_id,
            "migration",
            {"island": 0, "submission_count": 25},
        )
        assert not islands.migration_due(conn, run_id, 0)


def test_island_count_is_fixed(home: Path) -> None:
    run_id = _create_run(home, 5)
    with transaction(home) as conn:
        assert islands.island_count(conn, run_id) == 5
