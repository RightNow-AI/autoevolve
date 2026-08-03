"""Build deterministic SQLite runs for CLI visualization and report tests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

RUN_ID = "r0000000001"
METRIC = "quality"
DOMAIN = "synthetic-viz"

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


class ScriptedRng:
    """Small deterministic RNG whose sequence is visible in fixture source."""

    def __init__(self, values: Sequence[float] = (0.13, 0.61, 0.29, 0.83, 0.47)):
        self._values = tuple(values)
        self._index = 0

    def random(self) -> float:
        value = self._values[self._index % len(self._values)]
        self._index += 1
        return value

    def choice(self, values: Sequence[str]) -> str:
        index = int(self.random() * len(values)) % len(values)
        return values[index]


def build_fixture(
    db_path: Path,
    *,
    status: str = "budget_exhausted",
    run_id: str = RUN_ID,
) -> str:
    """Build the normative 40-program, 3-island visualization fixture."""

    _build(db_path, status=status, run_id=run_id, program_count=40)
    return run_id


def make_fixture(
    db_path: Path,
    *,
    status: str = "budget_exhausted",
    run_id: str = RUN_ID,
) -> str:
    """Compatibility entrypoint with the module's descriptive name."""

    return build_fixture(db_path, status=status, run_id=run_id)


def build_status_fixture(db_path: Path, status: str, *, run_id: str = RUN_ID) -> str:
    """Build a small terminal run for report end-state coverage."""

    program_count = 1 if status == "infeasible" else 4
    _build(db_path, status=status, run_id=run_id, program_count=program_count)
    return run_id


def _build(db_path: Path, *, status: str, run_id: str, program_count: int) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(DDL)
        rng = ScriptedRng()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        budget = {"max_cost_usd": None, "max_evals": program_count - 1, "wall_clock_s": None}
        feasibility: dict[str, Any] | None = None
        target = 3.0 if status == "target_hit" else 4.0
        if status == "target_hit" and program_count < 36:
            target = 1.2
        if status == "infeasible":
            target = 8.0
            feasibility = {"method": "scripted ceiling", "metric": METRIC, "value": 3.2}
        contract = {
            "baseline": 1.0,
            "budget": budget,
            "descriptors": [
                {"bins": 5, "hi": 1.0, "lo": 0.0, "metric": "novelty", "name": "novelty"}
            ],
            "domain": DOMAIN,
            "feasibility": feasibility,
            "gate": "correct",
            "goal": "Improve deterministic synthetic quality",
            "maximize": True,
            "metric": METRIC,
            "plateau_n": 2 if status == "plateau" and program_count < 40 else 12,
            "target": target,
        }
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                contract["goal"],
                DOMAIN,
                json.dumps(contract, sort_keys=True),
                status,
                json.dumps(budget, sort_keys=True),
                731,
                "tests/fixtures/viz",
                _timestamp(start, 0),
            ),
        )
        connection.executemany(
            "INSERT INTO islands VALUES (?, ?, ?, ?)",
            [(run_id, island, f"worker-{island}", None) for island in range(3)],
        )
        operators = ("diff", "rewrite", "agentic", "crossover")
        connection.executemany(
            "INSERT INTO operators VALUES (?, ?, ?, ?, ?)",
            [
                (DOMAIN, name, 10, 2 if name == "diff" else 1, 0.12 - index * 0.02)
                for index, name in enumerate(operators)
            ],
        )
        seed_id = _program_id(0)
        connection.execute(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (seed_id, run_id, None, "seed", "sha256:seed", 0, "0", _timestamp(start, 1)),
        )
        _insert_scores(connection, seed_id, 1.0, True, _timestamp(start, 1))
        sequence = 0

        def event(kind: str, payload: dict[str, Any], offset: int) -> None:
            nonlocal sequence
            sequence += 1
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    sequence,
                    kind,
                    json.dumps(payload, sort_keys=True),
                    _timestamp(start, offset),
                ),
            )

        event("run_opened", {"run_id": run_id}, 0)
        event("contract_locked", {"metric": METRIC}, 1)
        for island in range(3):
            event("worker_joined", {"island": island, "runtime": "fixture"}, 2 + island)

        gate_failures = {5, 11, 18, 25, 31, 38}
        improvements = {3: 1.2, 8: 1.5, 15: 1.9, 24: 2.4, 35: 3.0}
        if program_count < 40 and status == "plateau":
            improvements = {1: 1.1}
        elif program_count < 40 and status == "budget_exhausted":
            improvements = {2: 1.1}
        crossover_indexes = {14, 29}
        migration_index = 22
        best_value = 1.0
        generation_parents = {0: seed_id, 1: seed_id, 2: seed_id}
        child_index = 1
        for generation in range(1, 9):
            next_generation = dict(generation_parents)
            nodes_this_generation = min(5, program_count - child_index)
            for _ in range(nodes_this_generation):
                index = child_index
                island = (index + generation) % 3
                parent_id = generation_parents[island]
                program_id = _program_id(index)
                operator = operators[(index - 1) % len(operators)]
                created_offset = 10 + index * 4
                connection.execute(
                    "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        program_id,
                        run_id,
                        parent_id,
                        operator,
                        f"sha256:{index:064x}",
                        island,
                        f"{generation % 5}",
                        _timestamp(start, created_offset),
                    ),
                )
                connection.execute(
                    "INSERT INTO edges VALUES (?, ?, ?)",
                    (program_id, parent_id, "parent"),
                )
                if index in crossover_indexes:
                    other_island = (island + 1) % 3
                    connection.execute(
                        "INSERT INTO edges VALUES (?, ?, ?)",
                        (program_id, generation_parents[other_island], "crossover"),
                    )
                if index == migration_index:
                    other_island = (island + 2) % 3
                    connection.execute(
                        "INSERT INTO edges VALUES (?, ?, ?)",
                        (program_id, generation_parents[other_island], "migration"),
                    )
                passed = index not in gate_failures
                if index in improvements and passed:
                    score = improvements[index]
                    best_value = score
                elif passed:
                    score = max(0.05, best_value - 0.1 - rng.random() * 0.35)
                else:
                    score = 0.0
                _insert_scores(
                    connection,
                    program_id,
                    score,
                    passed,
                    _timestamp(start, created_offset + 1),
                )
                event(
                    "parent_sampled",
                    {"island": island, "parent_id": parent_id, "program_id": program_id},
                    created_offset,
                )
                submission_payload = {
                    "eval_idx": index,
                    "files_changed": 1 + index % 3,
                    "fitness": score,
                    "island": island,
                    "operator": operator,
                    "program_id": program_id,
                }
                event(
                    "child_submitted" if passed else "gate_failed",
                    submission_payload,
                    created_offset + 1,
                )
                event(
                    "operator_update",
                    {
                        "improved": index in improvements,
                        "operator": operator,
                        "program_id": program_id,
                    },
                    created_offset + 2,
                )
                if index in improvements:
                    event(
                        "archive_improved",
                        {"best_fitness": score, "eval_idx": index, "program_id": program_id},
                        created_offset + 2,
                    )
                if index == migration_index:
                    event(
                        "migration",
                        {
                            "from_island": (island + 2) % 3,
                            "program_id": program_id,
                            "to_island": island,
                        },
                        created_offset + 3,
                    )
                next_generation[island] = program_id
                child_index += 1
                if child_index >= program_count:
                    break
            generation_parents = next_generation
            if child_index >= program_count:
                break

        discoveries = [
            ("d0000000001", "Alternating operators improved exploration.", seed_id),
            (
                "d0000000002",
                "Island two produced the strongest late candidate.",
                _program_id(max(0, program_count - 1)),
            ),
        ]
        for discovery_index, (discovery_id, text, source_program) in enumerate(discoveries):
            created = _timestamp(start, 300 + discovery_index)
            connection.execute(
                "INSERT INTO discoveries VALUES (?, ?, ?, ?, ?, ?)",
                (discovery_id, DOMAIN, text, run_id, json.dumps([source_program]), created),
            )
            event(
                "discovery_added",
                {"discovery_id": discovery_id, "program_id": source_program},
                300 + discovery_index,
            )
        terminal_kind = {
            "budget": "budget_exhausted",
            "budget_exhausted": "budget_exhausted",
            "plateau": "plateau_detected",
            "target_hit": "target_hit",
        }.get(status)
        if terminal_kind is not None:
            event(terminal_kind, {"best_fitness": best_value}, 400)
        event("run_closed", {"reason": status}, 401)
        connection.commit()
    finally:
        connection.close()


def _insert_scores(
    connection: sqlite3.Connection,
    program_id: str,
    score: float,
    passed: bool,
    measured_at: str,
) -> None:
    connection.executemany(
        "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
        [
            (program_id, "correct", 1.0 if passed else 0.0, 0, measured_at),
            (program_id, METRIC, score, 0, measured_at),
        ],
    )


def _program_id(index: int) -> str:
    return f"p{index:010x}"


def _timestamp(start: datetime, seconds: int) -> str:
    return (start + timedelta(seconds=seconds)).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument(
        "--status",
        choices=("budget_exhausted", "infeasible", "plateau", "target_hit"),
        default="budget_exhausted",
    )
    arguments = parser.parse_args()
    build_fixture(arguments.db_path, status=arguments.status)
    print(arguments.db_path)


if __name__ == "__main__":
    main()
