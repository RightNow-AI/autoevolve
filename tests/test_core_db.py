"""Database and append-only event guarantees for the core engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoevolve.core.db import connection, init_db, new_id, transaction, utc_now
from autoevolve.core.events import EVENT_KINDS, append_event, load_events


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def _insert_run(home: Path, run_id: str) -> None:
    with transaction(home) as conn:
        conn.execute(
            "INSERT INTO runs(id, goal_text, domain, contract_json, status, budget_json, "
            "seed, evaluator_ref, created_at) VALUES (?, 'goal', 'domain', '{}', 'open', "
            "?, 7, NULL, ?)",
            (run_id, json.dumps({"max_evals": 2}), utc_now()),
        )


def test_schema_creation_is_idempotent_and_configures_sqlite(home: Path) -> None:
    database = init_db(home)
    assert init_db(home) == database
    with connection(home) as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "runs",
            "programs",
            "scores",
            "edges",
            "islands",
            "operators",
            "discoveries",
            "events",
        } <= tables
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 10_000
        assert int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) == 1


def test_events_are_gapless_monotonic_and_decoded(home: Path) -> None:
    init_db(home)
    run_id = new_id("r")
    _insert_run(home, run_id)
    with transaction(home) as conn:
        assert append_event(conn, run_id, "run_opened", {"seed": 7}) == 0
        assert append_event(conn, run_id, "contract_locked", {"metric": "score"}) == 1
        assert append_event(conn, run_id, "worker_joined", {"island": 0}) == 2
    with connection(home) as conn:
        events = load_events(conn, run_id)
    assert [event["seq"] for event in events] == [0, 1, 2]
    assert [event["kind"] for event in events] == [
        "run_opened",
        "contract_locked",
        "worker_joined",
    ]
    assert events[0]["payload"] == {"seed": 7}


def test_event_kind_set_is_closed(home: Path) -> None:
    init_db(home)
    run_id = new_id("r")
    _insert_run(home, run_id)
    assert "archive_improved" in EVENT_KINDS
    with transaction(home) as conn, pytest.raises(ValueError, match="unknown event kind"):
        append_event(conn, run_id, "surprise", {})


def test_ids_and_timestamps_have_public_shapes(home: Path) -> None:
    init_db(home)
    assert len(new_id("r")) == 11
    assert len(new_id("p")) == 11
    assert len(new_id("d")) == 11
    assert utc_now().endswith("+00:00")
