"""MAP-elites binning, fitness, and replacement tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoevolve.core import archive
from autoevolve.core.db import init_db, transaction, utc_now
from autoevolve.core.events import append_event
from autoevolve.core.types import Budget, Contract, Descriptor


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def _contract(descriptors: list[Descriptor] | None = None) -> Contract:
    return Contract(
        goal="score it",
        domain="test",
        metric="score",
        maximize=True,
        baseline=1.0,
        target=None,
        gate="correct",
        budget=Budget(max_evals=10),
        descriptors=descriptors or [],
    )


def _seed_archive(home: Path, fitness: float = 1.0) -> tuple[str, str]:
    init_db(home)
    run_id = "r0000000001"
    program_id = "p0000000001"
    contract = _contract()
    now = utc_now()
    with transaction(home) as conn:
        conn.execute(
            "INSERT INTO runs(id, goal_text, domain, contract_json, status, budget_json, "
            "seed, evaluator_ref, created_at) VALUES (?, ?, ?, ?, 'open', ?, 1, NULL, ?)",
            (
                run_id,
                contract.goal,
                contract.domain,
                contract.to_json(),
                json.dumps({"max_evals": 10}),
                now,
            ),
        )
        conn.execute(
            "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
            "cell_key, created_at) VALUES (?, ?, NULL, 'seed', 'ref', 0, '0', ?)",
            (program_id, run_id, now),
        )
        conn.execute(
            "INSERT INTO scores(program_id, metric, value, stage, measured_at) "
            "VALUES (?, 'score', ?, 0, ?)",
            (program_id, fitness, now),
        )
        append_event(
            conn,
            run_id,
            "archive_improved",
            {
                "program_id": program_id,
                "cell_key": "0",
                "fitness": fitness,
                "best_fitness": fitness,
                "eval_idx": 0,
            },
        )
    return run_id, program_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-5.0, "0"), (0.0, "0"), (10.0, "3"), (50.0, "3")],
)
def test_cell_binning_clamps_lo_hi_and_outside(
    home: Path,
    value: float,
    expected: str,
) -> None:
    descriptor = Descriptor(name="size", metric="size", bins=4, lo=0.0, hi=10.0)
    assert archive.cell_key(_contract([descriptor]), {"size": value}) == expected


def test_no_descriptors_use_single_cell(home: Path) -> None:
    assert archive.cell_key(_contract(), {"score": 3.0}) == "0"


def test_multiple_descriptors_join_bin_indices(home: Path) -> None:
    descriptors = [
        Descriptor(name="x", metric="x", bins=2, lo=0.0, hi=1.0),
        Descriptor(name="y", metric="y", bins=5, lo=0.0, hi=10.0),
    ]
    assert archive.cell_key(_contract(descriptors), {"x": 0.75, "y": 4.1}) == "1,2"


def test_archive_replacement_requires_strictly_greater_fitness(home: Path) -> None:
    run_id, program_id = _seed_archive(home, 2.0)
    with transaction(home) as conn:
        replace_equal, existing = archive.should_replace(conn, run_id, "0", 2.0)
        replace_lower, _ = archive.should_replace(conn, run_id, "0", 1.9)
        replace_higher, _ = archive.should_replace(conn, run_id, "0", 2.1)
    assert existing is not None and existing.program.id == program_id
    assert not replace_equal
    assert not replace_lower
    assert replace_higher


def test_minimize_contract_negates_fitness(home: Path) -> None:
    contract = _contract()
    contract.maximize = False
    assert archive.fitness(contract, 3.5) == -3.5
