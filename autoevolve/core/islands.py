"""Island creation, assignment, and migration cadence helpers."""

from __future__ import annotations

import json
import sqlite3

from autoevolve.core.db import utc_now

MIGRATION_INTERVAL = 25


def create_islands(conn: sqlite3.Connection, run_id: str, count: int) -> None:
    """Create the fixed island ring for a new run."""

    if count <= 0:
        raise ValueError("workers must be a positive integer")
    conn.executemany(
        "INSERT INTO islands(run_id, island_id, worker_hint, last_migration_at) "
        "VALUES (?, ?, NULL, NULL)",
        [(run_id, island_id) for island_id in range(count)],
    )


def island_count(conn: sqlite3.Connection, run_id: str) -> int:
    """Return the fixed number of islands for a run."""

    row = conn.execute(
        "SELECT COUNT(*) AS count FROM islands WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row["count"])


def assign_island(conn: sqlite3.Connection, run_id: str, runtime: str) -> int:
    """Assign workers round-robin using the durable worker-joined event count."""

    count = island_count(conn, run_id)
    if count == 0:
        raise KeyError(f"run has no islands: {run_id}")
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM events WHERE run_id = ? AND kind = 'worker_joined'",
        (run_id,),
    ).fetchone()
    assigned = int(row["count"]) % count
    conn.execute(
        "UPDATE islands SET worker_hint = ? WHERE run_id = ? AND island_id = ?",
        (runtime, run_id, assigned),
    )
    return assigned


def neighbor_island(conn: sqlite3.Connection, run_id: str, island: int) -> int:
    """Return the next island in ring order."""

    count = island_count(conn, run_id)
    if count == 0 or island < 0 or island >= count:
        raise ValueError(f"invalid island {island} for run {run_id}")
    return (island + 1) % count


def submission_count(conn: sqlite3.Connection, run_id: str, island: int) -> int:
    """Count evaluated child programs assigned to one island."""

    row = conn.execute(
        "SELECT COUNT(*) AS count FROM programs "
        "WHERE run_id = ? AND island = ? AND parent_id IS NOT NULL",
        (run_id, island),
    ).fetchone()
    return int(row["count"])


def migration_due(conn: sqlite3.Connection, run_id: str, island: int) -> bool:
    """Return whether the next sample is the once-per-25-submissions migration slot."""

    submissions = submission_count(conn, run_id, island)
    if submissions == 0 or submissions % MIGRATION_INTERVAL != 0:
        return False
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE run_id = ? AND kind = 'migration'",
        (run_id,),
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        if int(payload.get("island", -1)) == island and int(
            payload.get("submission_count", -1)
        ) == submissions:
            return False
    return True


def mark_migration(conn: sqlite3.Connection, run_id: str, island: int) -> None:
    """Record migration metadata for operators and status displays."""

    conn.execute(
        "UPDATE islands SET last_migration_at = ? WHERE run_id = ? AND island_id = ?",
        (utc_now(), run_id, island),
    )
