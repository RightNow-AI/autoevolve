"""Validated append-only event operations."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from autoevolve.core.db import utc_now

EVENT_KINDS = frozenset(
    {
        "run_opened",
        "contract_locked",
        "worker_joined",
        "parent_sampled",
        "child_submitted",
        "gate_failed",
        "archive_improved",
        "migration",
        "operator_update",
        "discovery_added",
        "plateau_detected",
        "target_hit",
        "budget_exhausted",
        "run_closed",
    }
)


def next_sequence(conn: sqlite3.Connection, run_id: str) -> int:
    """Return the next gapless sequence number for a run."""

    row = conn.execute(
        "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM events WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row["next_seq"])


def append_event(
    conn: sqlite3.Connection,
    run_id: str,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> int:
    """Append one validated event and return its assigned sequence number."""

    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind: {kind}")
    seq = next_sequence(conn, run_id)
    conn.execute(
        "INSERT INTO events(run_id, seq, kind, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            run_id,
            seq,
            kind,
            json.dumps(payload or {}, sort_keys=True, separators=(",", ":")),
            utc_now(),
        ),
    )
    return seq


def load_events(
    conn: sqlite3.Connection,
    run_id: str,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Return decoded events in sequence order."""

    if kind is None:
        rows = conn.execute(
            "SELECT seq, kind, payload_json, created_at FROM events "
            "WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
    else:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        rows = conn.execute(
            "SELECT seq, kind, payload_json, created_at FROM events "
            "WHERE run_id = ? AND kind = ? ORDER BY seq",
            (run_id, kind),
        ).fetchall()
    return [
        {
            "seq": int(row["seq"]),
            "kind": str(row["kind"]),
            "payload": json.loads(row["payload_json"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]
