"""MAP-elites cell computation and event-backed elite queries."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from autoevolve.core.types import Contract, Descriptor, Program


@dataclass(frozen=True)
class Elite:
    """The current best program recorded for one archive cell."""

    program: Program
    scores: dict[str, float]
    fitness: float
    cell_key: str


def descriptor_bin(descriptor: Descriptor, value: float) -> int:
    """Clamp a descriptor value to its configured zero-based bin."""

    if descriptor.bins <= 0:
        raise ValueError(f"descriptor {descriptor.name!r} must have positive bins")
    if descriptor.hi <= descriptor.lo:
        raise ValueError(f"descriptor {descriptor.name!r} must have hi greater than lo")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"descriptor {descriptor.name!r} must be finite")
    clamped = min(max(numeric, descriptor.lo), descriptor.hi)
    if clamped == descriptor.hi:
        return descriptor.bins - 1
    fraction = (clamped - descriptor.lo) / (descriptor.hi - descriptor.lo)
    return min(int(fraction * descriptor.bins), descriptor.bins - 1)


def cell_key(contract: Contract, scores: dict[str, float]) -> str:
    """Return the MAP-elites cell key for a score mapping."""

    if not contract.descriptors:
        return "0"
    bins: list[str] = []
    for descriptor in contract.descriptors:
        if descriptor.metric not in scores:
            raise ValueError(
                f"descriptor metric {descriptor.metric!r} missing from evaluator scores"
            )
        bins.append(str(descriptor_bin(descriptor, scores[descriptor.metric])))
    return ",".join(bins)


def fitness(contract: Contract, metric_value: float) -> float:
    """Normalize maximize and minimize contracts so larger is always better."""

    value = float(metric_value)
    if not math.isfinite(value):
        raise ValueError("contract metric must be finite")
    return value if contract.maximize else -value


def program_from_row(row: sqlite3.Row) -> Program:
    """Convert a programs row into the shared immutable type."""

    return Program(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
        operator=str(row["operator"]),
        code_ref=str(row["code_ref"]),
        island=int(row["island"]),
        cell_key=str(row["cell_key"]) if row["cell_key"] is not None else None,
        created_at=str(row["created_at"]),
    )


def scores_for_program(conn: sqlite3.Connection, program_id: str) -> dict[str, float]:
    """Return the deepest recorded value for every program metric."""

    rows = conn.execute(
        "SELECT metric, value, stage FROM scores WHERE program_id = ? "
        "ORDER BY metric, stage",
        (program_id,),
    ).fetchall()
    scores: dict[str, float] = {}
    for row in rows:
        scores[str(row["metric"])] = float(row["value"])
    return scores


def _latest_cell_payloads(
    conn: sqlite3.Connection,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE run_id = ? AND kind = 'archive_improved' "
        "ORDER BY seq",
        (run_id,),
    ).fetchall()
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = json.loads(row["payload_json"])
        cells[str(payload["cell_key"])] = payload
    return cells


def current_elites(conn: sqlite3.Connection, run_id: str) -> list[Elite]:
    """Reconstruct the current one-program-per-cell archive from events."""

    elites: list[Elite] = []
    for key, payload in _latest_cell_payloads(conn, run_id).items():
        row = conn.execute(
            "SELECT * FROM programs WHERE id = ? AND run_id = ?",
            (payload["program_id"], run_id),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"archive event references missing program {payload['program_id']}")
        elites.append(
            Elite(
                program=program_from_row(row),
                scores=scores_for_program(conn, str(payload["program_id"])),
                fitness=float(payload["fitness"]),
                cell_key=key,
            )
        )
    return sorted(elites, key=lambda item: item.cell_key)


def elite_for_cell(
    conn: sqlite3.Connection,
    run_id: str,
    key: str,
) -> Elite | None:
    """Return the current elite for a cell, if one has been recorded."""

    return next((elite for elite in current_elites(conn, run_id) if elite.cell_key == key), None)


def should_replace(
    conn: sqlite3.Connection,
    run_id: str,
    key: str,
    candidate_fitness: float,
) -> tuple[bool, Elite | None]:
    """Apply the strict-greater archive replacement rule without writing an event."""

    existing = elite_for_cell(conn, run_id, key)
    return existing is None or candidate_fitness > existing.fitness, existing


def best_elite(conn: sqlite3.Connection, run_id: str) -> Elite | None:
    """Return the highest-fitness current elite."""

    elites = current_elites(conn, run_id)
    if not elites:
        return None
    return min(elites, key=lambda item: (-item.fitness, item.program.id))
