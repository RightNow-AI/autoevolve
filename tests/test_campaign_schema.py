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

