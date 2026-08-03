"""Persistent per-domain UCB1 operator selection."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

OPERATORS = ("agentic", "crossover", "diff", "rewrite")


@dataclass(frozen=True)
class OperatorState:
    """One persisted operator arm."""

    domain: str
    name: str
    pulls: int
    improvements: int
    mean_gain: float


def ensure_operators(
    conn: sqlite3.Connection,
    domain: str,
    names: tuple[str, ...] = OPERATORS,
) -> None:
    """Create missing bandit arms without resetting learned state."""

    conn.executemany(
        "INSERT OR IGNORE INTO operators(domain, name, pulls, improvements, mean_gain) "
        "VALUES (?, ?, 0, 0, 0)",
        [(domain, name) for name in names],
    )


def states(
    conn: sqlite3.Connection,
    domain: str,
    names: tuple[str, ...] = OPERATORS,
) -> list[OperatorState]:
    """Return requested arms in name order."""

    ensure_operators(conn, domain, names)
    rows = conn.execute(
        "SELECT domain, name, pulls, improvements, mean_gain FROM operators "
        "WHERE domain = ? ORDER BY name",
        (domain,),
    ).fetchall()
    allowed = set(names)
    return [
        OperatorState(
            domain=str(row["domain"]),
            name=str(row["name"]),
            pulls=int(row["pulls"]),
            improvements=int(row["improvements"]),
            mean_gain=float(row["mean_gain"]),
        )
        for row in rows
        if str(row["name"]) in allowed
    ]


def select_operator(
    conn: sqlite3.Connection,
    domain: str,
    names: tuple[str, ...] = OPERATORS,
) -> str:
    """Select an unpulled arm first, otherwise maximize the UCB1 score."""

    arms = states(conn, domain, names)
    unpulled = sorted(arm.name for arm in arms if arm.pulls == 0)
    if unpulled:
        return unpulled[0]
    total_pulls = sum(arm.pulls for arm in arms)

    def ucb(arm: OperatorState) -> float:
        return arm.mean_gain + math.sqrt(2.0 * math.log(total_pulls) / arm.pulls)

    return min(arms, key=lambda arm: (-ucb(arm), arm.name)).name


def relative_gain(parent_fitness: float, child_fitness: float) -> float:
    """Return the clipped relative fitness gain used as the bandit reward."""

    denominator = max(abs(parent_fitness), 1e-9)
    raw = (child_fitness - parent_fitness) / denominator
    return min(max(raw, -1.0), 1.0)


def update_operator(
    conn: sqlite3.Connection,
    domain: str,
    name: str,
    parent_fitness: float,
    child_fitness: float,
) -> OperatorState:
    """Apply one incremental reward update and return the persisted arm."""

    ensure_operators(conn, domain, (name,))
    row = conn.execute(
        "SELECT pulls, improvements, mean_gain FROM operators "
        "WHERE domain = ? AND name = ?",
        (domain, name),
    ).fetchone()
    pulls = int(row["pulls"])
    improvements = int(row["improvements"])
    mean_gain = float(row["mean_gain"])
    gain = relative_gain(parent_fitness, child_fitness)
    next_pulls = pulls + 1
    next_improvements = improvements + int(gain > 0)
    next_mean = mean_gain + (gain - mean_gain) / next_pulls
    conn.execute(
        "UPDATE operators SET pulls = ?, improvements = ?, mean_gain = ? "
        "WHERE domain = ? AND name = ?",
        (next_pulls, next_improvements, next_mean, domain, name),
    )
    return OperatorState(domain, name, next_pulls, next_improvements, next_mean)
