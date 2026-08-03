import json
import sqlite3
from pathlib import Path

from autoevolve.mutate.distiller import distill_run
from tests.fakes import FakeEndpoint

DDL = """
CREATE TABLE runs(
  id TEXT PRIMARY KEY, goal_text TEXT NOT NULL, domain TEXT NOT NULL,
  contract_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  budget_json TEXT NOT NULL, seed INTEGER NOT NULL, evaluator_ref TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE programs(
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
  parent_id TEXT REFERENCES programs(id), operator TEXT NOT NULL,
  code_ref TEXT NOT NULL, island INTEGER NOT NULL, cell_key TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE scores(
  program_id TEXT NOT NULL REFERENCES programs(id), metric TEXT NOT NULL,
  value REAL NOT NULL, stage INTEGER NOT NULL, measured_at TEXT NOT NULL,
  PRIMARY KEY(program_id, metric, stage));
CREATE TABLE edges(
  child_id TEXT NOT NULL, parent_id TEXT NOT NULL, kind TEXT NOT NULL,
  PRIMARY KEY(child_id, parent_id, kind));
CREATE TABLE islands(
  run_id TEXT NOT NULL, island_id INTEGER NOT NULL, worker_hint TEXT,
  last_migration_at TEXT, PRIMARY KEY(run_id, island_id));
CREATE TABLE operators(
  domain TEXT NOT NULL, name TEXT NOT NULL, pulls INTEGER NOT NULL DEFAULT 0,
  improvements INTEGER NOT NULL DEFAULT 0, mean_gain REAL NOT NULL DEFAULT 0,
  PRIMARY KEY(domain, name));
CREATE TABLE discoveries(
  id TEXT PRIMARY KEY, domain TEXT NOT NULL, text TEXT NOT NULL,
  source_run TEXT, source_programs TEXT, created_at TEXT NOT NULL);
CREATE TABLE events(
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, seq));
CREATE INDEX idx_programs_run ON programs(run_id);
CREATE INDEX idx_scores_program ON scores(program_id);
"""


def test_distill_run_persists_statements_events_and_markdown(tmp_path):
    _build_run(tmp_path)
    endpoint = FakeEndpoint(
        [
            """- Raising the constant from 1 to 4 improved the measured score from 1.0 to 4.0.
- Keeping the frozen wrapper unchanged preserved the deterministic correctness gate.
- Concentrating the change in one marked assignment produced the observed gain with less risk.
"""
        ]
    )

    rows = distill_run(tmp_path, "r1", endpoint, top_k=1)

    assert len(rows) == 3
    assert all(row["source_programs"] == ["pchild"] for row in rows)
    with sqlite3.connect(tmp_path / "autoevolve.db") as connection:
        discoveries = connection.execute(
            "SELECT id, domain, source_run, source_programs FROM discoveries ORDER BY id"
        ).fetchall()
        events = connection.execute(
            "SELECT seq, kind, payload_json FROM events ORDER BY seq"
        ).fetchall()
    assert len(discoveries) == 3
    assert all(row[1:3] == ("python-speedup", "r1") for row in discoveries)
    assert all(json.loads(row[3]) == ["pchild"] for row in discoveries)
    assert [row[0] for row in events] == [4, 5, 6, 7]
    assert [row[1] for row in events[1:]] == ["discovery_added"] * 3
    assert all(json.loads(row[2])["source_programs"] == ["pchild"] for row in events[1:])

    mirror = (tmp_path / "discoveries" / "python-speedup.md").read_text(encoding="utf-8")
    assert "run r1" in mirror
    assert "pchild" in mirror
    assert rows[0]["text"] in mirror
    prompt = endpoint.calls[0][1]["content"]
    assert "pbase/main.py" in prompt
    assert "pchild/main.py" in prompt


def test_distill_run_without_endpoint_writes_nothing(tmp_path):
    assert distill_run(tmp_path, "missing", None) == []
    assert not (tmp_path / "discoveries").exists()
    assert not (tmp_path / "autoevolve.db").exists()


def _build_run(home: Path) -> None:
    database = home / "autoevolve.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(DDL)
        contract = json.dumps({"metric": "score", "gate": "correct", "maximize": True})
        connection.execute(
            """
            INSERT INTO runs(
                id, goal_text, domain, contract_json, budget_json, seed, created_at
            ) VALUES ('r1', 'make Python faster', 'python-speedup', ?, '{}', 1, 'now')
            """,
            (contract,),
        )
        connection.executemany(
            """
            INSERT INTO programs(
                id, run_id, parent_id, operator, code_ref, island, cell_key, created_at
            ) VALUES (?, 'r1', ?, ?, ?, 0, '0', 'now')
            """,
            [
                ("pbase", None, "seed", "base-ref"),
                ("pchild", "pbase", "diff", "child-ref"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO scores(program_id, metric, value, stage, measured_at)
            VALUES (?, ?, ?, 0, 'now')
            """,
            [
                ("pbase", "score", 1.0),
                ("pbase", "correct", 1.0),
                ("pchild", "score", 4.0),
                ("pchild", "correct", 1.0),
            ],
        )
        connection.execute(
            "INSERT INTO events VALUES ('r1', 4, 'archive_improved', '{}', 'now')"
        )

    base = home / "store" / "base-ref"
    child = home / "store" / "child-ref"
    base.mkdir(parents=True)
    child.mkdir(parents=True)
    base.joinpath("main.py").write_text("value = 1\n", encoding="utf-8")
    child.joinpath("main.py").write_text("value = 4\n", encoding="utf-8")
