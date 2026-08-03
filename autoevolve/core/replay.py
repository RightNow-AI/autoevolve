"""Deterministic verification of recorded sampling and migration choices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoevolve.core import sampling
from autoevolve.core.db import connection, resolve_home
from autoevolve.core.events import load_events


def _assert_program_pool(conn: Any, run_id: str, candidate_ids: list[str]) -> None:
    for program_id in candidate_ids:
        row = conn.execute(
            "SELECT id FROM programs WHERE id = ? AND run_id = ?",
            (program_id, run_id),
        ).fetchone()
        assert row is not None, f"recorded replay pool references missing program {program_id}"


def _historical_elites(conn: Any, run_id: str, event_seq: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE run_id = ? "
        "AND kind = 'archive_improved' AND seq < ? ORDER BY seq",
        (run_id, event_seq),
    ).fetchall()
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = json.loads(row["payload_json"])
        cells[str(payload["cell_key"])] = payload
    elites: list[dict[str, Any]] = []
    for payload in cells.values():
        program = conn.execute(
            "SELECT id, island FROM programs WHERE id = ? AND run_id = ?",
            (payload["program_id"], run_id),
        ).fetchone()
        assert program is not None, "historical archive references a missing program"
        elites.append(
            {
                "program_id": str(program["id"]),
                "island": int(program["island"]),
                "fitness": float(payload["fitness"]),
            }
        )
    return sorted(elites, key=lambda item: (-item["fitness"], item["program_id"]))


def _replay_choice(
    conn: Any,
    run_id: str,
    run_seed: int,
    event_seq: int,
    payload: dict[str, Any],
) -> None:
    candidate_ids = [str(item) for item in payload["candidate_ids"]]
    weights = [float(item) for item in payload["weights"]]
    assert len(candidate_ids) == len(weights), "replay pool and weights differ in length"
    expected_weights = [float(weight) for weight in range(len(weights), 0, -1)]
    assert weights == expected_weights, "rank weights diverged"
    _assert_program_pool(conn, run_id, candidate_ids)
    historical = _historical_elites(conn, run_id, event_seq)
    global_ids = [str(item["program_id"]) for item in historical]
    if payload["rng_kind"] == "parent_sample":
        island = int(payload["island"])
        local_ids = [
            str(item["program_id"]) for item in historical if int(item["island"]) == island
        ]
        assert payload["global_candidate_ids"] == global_ids, "global replay pool diverged"
        assert payload["local_candidate_ids"] == local_ids, "local replay pool diverged"
        expected_pool = local_ids if payload["source"] == "local" else global_ids
    else:
        neighbor = payload.get("neighbor_island", payload.get("migration_from"))
        expected_pool = [
            str(item["program_id"])
            for item in historical
            if int(item["island"]) == int(neighbor)
        ]
    assert candidate_ids == expected_pool, "recorded candidate pool diverged"
    rng = sampling.seeded_rng(
        run_seed,
        str(payload["rng_kind"]),
        int(payload["rng_event_seq"]),
    )
    if payload["rng_kind"] == "parent_sample" and payload.get("local_candidate_ids"):
        branch_draw = rng.random()
        assert branch_draw == float(payload["branch_draw"]), "parent source draw diverged"
        expected_source = (
            "local"
            if branch_draw < float(payload["local_probability"])
            else "global"
        )
        assert expected_source == payload["source"], "parent source branch diverged"
    index = sampling.weighted_index(rng, weights)
    assert index == int(payload["chosen_index"]), "weighted parent index diverged"
    assert candidate_ids[index] == payload["chosen_parent_id"], "parent id diverged"


def replay(home: Path | None, run_id: str) -> dict[str, int | str]:
    """Re-derive every recorded parent and migration draw and assert equality."""

    resolved = resolve_home(home)
    with connection(resolved) as conn:
        run = conn.execute("SELECT seed FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        events = load_events(conn, run_id)
        expected_sequences = list(range(len(events)))
        actual_sequences = [int(event["seq"]) for event in events]
        assert actual_sequences == expected_sequences, "event sequence is not gapless"
        sampling_checks = 0
        migration_checks = 0
        for event in events:
            if event["kind"] == "parent_sampled":
                _replay_choice(
                    conn,
                    run_id,
                    int(run["seed"]),
                    int(event["seq"]),
                    event["payload"],
                )
                sampling_checks += 1
            elif event["kind"] == "migration":
                _replay_choice(
                    conn,
                    run_id,
                    int(run["seed"]),
                    int(event["seq"]),
                    event["payload"],
                )
                migration_checks += 1
    return {
        "run_id": run_id,
        "sampling_checks": sampling_checks,
        "migration_checks": migration_checks,
        "event_sequence_checks": len(events),
        "total_checks": sampling_checks + migration_checks + len(events),
    }
