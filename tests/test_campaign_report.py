from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from autoevolve.cli.campaign import (
    CampaignCell,
    CampaignConfig,
    build_campaign_report,
)
from autoevolve.core.db import SCHEMA


def _config(tmp_path: Path, name: str, cells: tuple[str, ...]) -> CampaignConfig:
    pack = tmp_path / name
    evaluator = pack / "evaluator"
    evaluator.mkdir(parents=True)
    return CampaignConfig(
        pack_dir=pack,
        name=name,
        domain="test-domain",
        evaluator="evaluator",
        cells=tuple(
            CampaignCell(cell, {"AUTOEVOLVE_CELL": cell}, None) for cell in cells
        ),
        proxy_budget={"max_evals": 5},
        full_budget={"max_evals": 50},
        ladder=("proxy", "replicate-3", "scaled"),
        replicate_seeds=3,
    )


def _database(home: Path) -> sqlite3.Connection:
    home.mkdir(parents=True)
    connection = sqlite3.connect(home / "autoevolve.db")
    connection.executescript(SCHEMA)
    return connection


def _insert_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    campaign_name: str,
    cell: str,
    seed: int,
    best: float,
    baseline: float = 1.0,
    r2_heldout: float | None = None,
) -> None:
    contract = {
        "baseline": baseline,
        "budget": {"max_evals": 5},
        "descriptors": [],
        "domain": "test-domain",
        "feasibility": None,
        "gate": "valid",
        "goal": f"campaign:{campaign_name}:{cell}",
        "maximize": True,
        "metric": "fitness",
        "plateau_n": 150,
        "target": None,
    }
    connection.execute(
        "INSERT INTO runs(id, goal_text, domain, contract_json, status, budget_json, "
        "seed, evaluator_ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            f"campaign:{campaign_name}:{cell}",
            "test-domain",
            json.dumps(contract),
            "budget_exhausted",
            json.dumps({"max_evals": 5}),
            seed,
            "evaluator",
            f"2026-08-03T00:00:{seed:02d}+00:00",
        ),
    )
    seed_program = f"p{run_id[1:]}s"
    best_program = f"p{run_id[1:]}b"
    connection.executemany(
        "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
        "cell_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                seed_program,
                run_id,
                None,
                "seed",
                "seed-ref",
                0,
                "0",
                "2026-08-03T00:00:00+00:00",
            ),
            (
                best_program,
                run_id,
                seed_program,
                "diff",
                "best-ref",
                0,
                "0",
                "2026-08-03T00:00:01+00:00",
            ),
        ],
    )
    score_rows = [
        (seed_program, "fitness", baseline, 0, "2026-08-03T00:00:00+00:00"),
        (seed_program, "valid", 1.0, 0, "2026-08-03T00:00:00+00:00"),
        (best_program, "fitness", best, 0, "2026-08-03T00:00:01+00:00"),
        (best_program, "valid", 1.0, 0, "2026-08-03T00:00:01+00:00"),
    ]
    if r2_heldout is not None:
        score_rows.append(
            (
                best_program,
                "r2_heldout",
                r2_heldout,
                0,
                "2026-08-03T00:00:01+00:00",
            )
        )
    connection.executemany(
        "INSERT INTO scores(program_id, metric, value, stage, measured_at) "
        "VALUES (?, ?, ?, ?, ?)",
        score_rows,
    )


def test_report_labels_proxy_candidate_and_replicated_cell(tmp_path: Path) -> None:
    home = tmp_path / "home"
    connection = _database(home)
    _insert_run(
        connection,
        run_id="r0000000001",
        campaign_name="report-pack",
        cell="candidate",
        seed=1,
        best=1.2,
    )
    for seed in (2, 3, 4):
        _insert_run(
            connection,
            run_id=f"r000000000{seed}",
            campaign_name="report-pack",
            cell="replicated",
            seed=seed,
            best=1.1 + seed / 10.0,
        )
    connection.commit()
    connection.close()
    config = _config(tmp_path, "report-pack", ("candidate", "replicated"))

    report = build_campaign_report(home, config, claims_root=None)

    assert "candidate | proxy candidate" in report
    assert "discovery | replicate-3" in report
    assert "requires an explicit scaled validation run" in report
    assert "not run or claimed automatically" in report


def test_equation_report_labels_high_heldout_fit_as_rediscovery(tmp_path: Path) -> None:
    home = tmp_path / "home"
    connection = _database(home)
    _insert_run(
        connection,
        run_id="r0000000009",
        campaign_name="equation-discovery",
        cell="nguyen-5",
        seed=9,
        best=1.05,
        r2_heldout=0.995,
    )
    connection.commit()
    connection.close()
    config = _config(tmp_path, "equation-discovery", ("nguyen-5",))

    report = build_campaign_report(home, config, claims_root=None)

    assert "rediscovery" in report
    assert "`r0000000009`" in report
