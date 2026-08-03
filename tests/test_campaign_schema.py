from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoevolve.cli.campaign import (
    CAMPAIGNS_ROOT,
    CampaignError,
    discover_campaigns,
    load_campaign,
)


def test_all_four_campaign_configs_parse_and_validate() -> None:
    configs = discover_campaigns(CAMPAIGNS_ROOT)

    assert {config.name for config in configs} == {
        "algorithm-frontier",
        "arch-search",
        "equation-discovery",
        "kernel-frontier",
    }
    for config in configs:
        assert config.cells
        assert config.ladder
        assert config.budget(full=False).is_bounded()
        assert config.budget(full=True).is_bounded()
        assert config.evaluator_path.is_dir()


def test_malformed_campaign_config_has_a_clear_error(tmp_path: Path) -> None:
    pack = tmp_path / "broken"
    pack.mkdir()
    (pack / "campaign.json").write_text(
        json.dumps({"name": "broken", "domain": "test"}),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="missing required keys"):
        load_campaign(pack)



def _write_bounds(pack: Path, checked_on: str) -> None:
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "bounds.json").write_text(
        json.dumps(
            {
                "bounds": [
                    {
                        "claim": "smallest known 5-chromatic unit-distance graph",
                        "value": "509",
                        "direction": "lower_is_better",
                        "who_and_year": "Parts, 2020",
                        "source_url": "https://example.invalid/parts",
                        "checked_on": checked_on,
                        "how_to_recheck": "re-read the tracked reduction list",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_bounds_staleness_is_measured_from_the_recheck_date(tmp_path: Path) -> None:
    from datetime import date, timedelta

    from autoevolve.cli.campaign import STALE_AFTER_DAYS, load_bounds

    today = date(2026, 8, 3)
    fresh_pack = tmp_path / "fresh"
    _write_bounds(fresh_pack, (today - timedelta(days=3)).isoformat())
    stale_pack = tmp_path / "stale"
    _write_bounds(stale_pack, (today - timedelta(days=STALE_AFTER_DAYS + 5)).isoformat())

    fresh = load_bounds(fresh_pack)[0]
    stale = load_bounds(stale_pack)[0]

    assert fresh.age_days(today) == 3
    assert not fresh.is_stale(today)
    assert stale.is_stale(today)


def test_bounds_reject_entries_missing_their_citation(tmp_path: Path) -> None:
    from autoevolve.cli.campaign import CampaignError, load_bounds

    pack = tmp_path / "bad"
    pack.mkdir()
    (pack / "bounds.json").write_text(
        json.dumps({"bounds": [{"claim": "x", "value": "1"}]}), encoding="utf-8"
    )

    with pytest.raises(CampaignError, match="missing required fields"):
        load_bounds(pack)


def test_packs_without_bounds_are_allowed(tmp_path: Path) -> None:
    from autoevolve.cli.campaign import load_bounds

    assert load_bounds(tmp_path) == ()
