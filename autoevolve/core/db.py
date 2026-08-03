"""SQLite setup and transaction helpers for the global evolution store."""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_WRITE_LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, goal_text TEXT NOT NULL, domain TEXT NOT NULL,
  contract_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  budget_json TEXT NOT NULL, seed INTEGER NOT NULL, evaluator_ref TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS programs(
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
  parent_id TEXT REFERENCES programs(id), operator TEXT NOT NULL,
  code_ref TEXT NOT NULL, island INTEGER NOT NULL, cell_key TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scores(
  program_id TEXT NOT NULL REFERENCES programs(id), metric TEXT NOT NULL,
  value REAL NOT NULL, stage INTEGER NOT NULL, measured_at TEXT NOT NULL,
  PRIMARY KEY(program_id, metric, stage));
CREATE TABLE IF NOT EXISTS edges(
  child_id TEXT NOT NULL, parent_id TEXT NOT NULL, kind TEXT NOT NULL,
  PRIMARY KEY(child_id, parent_id, kind));
CREATE TABLE IF NOT EXISTS islands(
  run_id TEXT NOT NULL, island_id INTEGER NOT NULL, worker_hint TEXT,
  last_migration_at TEXT, PRIMARY KEY(run_id, island_id));
CREATE TABLE IF NOT EXISTS operators(
  domain TEXT NOT NULL, name TEXT NOT NULL, pulls INTEGER NOT NULL DEFAULT 0,
  improvements INTEGER NOT NULL DEFAULT 0, mean_gain REAL NOT NULL DEFAULT 0,
  PRIMARY KEY(domain, name));
CREATE TABLE IF NOT EXISTS discoveries(
  id TEXT PRIMARY KEY, domain TEXT NOT NULL, text TEXT NOT NULL,
  source_run TEXT, source_programs TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, seq));
CREATE INDEX IF NOT EXISTS idx_programs_run ON programs(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_program ON scores(program_id);
"""


def resolve_home(home: Path | None = None) -> Path:
    """Return the configured AutoEvolve home and ensure it exists."""

    if home is None:
        configured = os.environ.get("AUTOEVOLVE_HOME")
        home = Path(configured) if configured else Path.home() / ".autoevolve"
    resolved = Path(home).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def database_path(home: Path | None = None) -> Path:
    """Return the SQLite file path for a home directory."""

    return resolve_home(home) / "autoevolve.db"


def _open(home: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path(home), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection(home: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a configured read connection and always close it."""

    conn = _open(home)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(home: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a serialized write transaction and commit it atomically."""

    with _WRITE_LOCK:
        conn = _open(home)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db(home: Path | None = None) -> Path:
    """Create the exact U1 schema if needed and return the database path."""

    resolved = resolve_home(home)
    with _WRITE_LOCK:
        conn = _open(resolved)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
    return database_path(resolved)


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 form."""

    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    """Return a compact public identifier with the requested prefix."""

    if prefix not in {"r", "p", "d"}:
        raise ValueError(f"unsupported id prefix: {prefix}")
    return prefix + uuid.uuid4().hex[:10]
