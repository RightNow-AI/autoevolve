from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from autoevolve.cli import campaign
from autoevolve.cli.campaign import CampaignCell, CampaignConfig


def _config(tmp_path: Path) -> CampaignConfig:
    pack = tmp_path / "runner-pack"
    evaluator = pack / "evaluator"
    evaluator.mkdir(parents=True)
    (pack / "log.md").write_text(
        "# Test campaign log\n\nThis file is append-only.\n",
        encoding="utf-8",
    )
    return CampaignConfig(
        pack_dir=pack,
        name="runner-pack",
        domain="test-domain",
        evaluator="evaluator",
        cells=(
            CampaignCell("first", {"AUTOEVOLVE_CELL": "first"}, None),
            CampaignCell("second", {"AUTOEVOLVE_CELL": "second"}, None),
        ),
        proxy_budget={"max_evals": 5},
        full_budget={"max_evals": 50},
        ladder=("proxy", "replicate-3", "scaled"),
        replicate_seeds=3,
    )


def test_list_discovers_every_campaign() -> None:
    assert {item.name for item in campaign.discover_campaigns()} >= {
        "algorithm-frontier",
        "arch-search",
        "equation-discovery",
        "kernel-frontier",
    }


def test_proxy_run_opens_each_cell_with_tag_budget_seed_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    counter = iter(("r0000000001", "r0000000002"))

    class FakeEngine:
        def __init__(self, home: Path):
            self.home = home

        def open_run(self, **kwargs: Any) -> dict[str, str]:
            calls.append(("open", (kwargs, campaign.os.environ.get("AUTOEVOLVE_CELL"))))
            return {"run_id": next(counter)}

        def run_status(self, run_id: str) -> dict[str, str]:
            return {"status": "budget_exhausted"}

        def best(self, run_id: str, k: int) -> list[dict[str, float]]:
            assert k == 1
            return [{"fitness": 1.25}]

    def fake_loop(engine: Any, run_id: str, get_operator: Any) -> None:
        calls.append(("loop", (engine, run_id, get_operator)))

    monkeypatch.setattr(campaign, "Engine", FakeEngine)
    monkeypatch.setattr(campaign, "run_worker_loop", fake_loop)
    config = _config(tmp_path)

    results = campaign.execute_campaign(
        config,
        cell_key=None,
        full=False,
        seed=17,
        home=tmp_path / "home",
    )

    open_calls = [value for name, value in calls if name == "open"]
    assert len(open_calls) == 2
    assert [item[0]["goal_text"] for item in open_calls] == [
        "campaign:runner-pack:first",
        "campaign:runner-pack:second",
    ]
    assert [item[0]["budget"].max_evals for item in open_calls] == [5, 5]
    assert [item[0]["seed"] for item in open_calls] == [17, 17]
    assert [item[1] for item in open_calls] == ["first", "second"]
    assert [result.cell for result in results] == ["first", "second"]

    log_text = (config.pack_dir / "log.md").read_text(encoding="utf-8")
    assert re.search(r"## \d{4}-\d{2}-\d{2}T", log_text)
    assert "Run id: `r0000000001`" in log_text
    assert "Cell: `first`" in log_text
    assert 'Budget: `{"max_evals":5}`' in log_text
    assert "Best fitness: 1.25" in log_text
    assert "End cause: `budget_exhausted`" in log_text


def test_cell_filter_and_full_budget_select_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[dict[str, Any]] = []

    class FakeEngine:
        def __init__(self, home: Path):
            self.home = home

        def open_run(self, **kwargs: Any) -> dict[str, str]:
            opened.append(kwargs)
            return {"run_id": "r0000000003", "status": "target_hit"}

        def run_status(self, run_id: str) -> dict[str, str]:
            return {"status": "target_hit"}

        def best(self, run_id: str, k: int) -> list[dict[str, float]]:
            return [{"fitness": 2.0}]

    monkeypatch.setattr(campaign, "Engine", FakeEngine)
    config = _config(tmp_path)

    campaign.execute_campaign(
        config,
        cell_key="second",
        full=True,
        seed=23,
        home=tmp_path / "home",
    )

    assert len(opened) == 1
    assert opened[0]["goal_text"] == "campaign:runner-pack:second"
    assert opened[0]["budget"].max_evals == 50
    assert opened[0]["seed"] == 23

