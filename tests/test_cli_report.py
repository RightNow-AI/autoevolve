from __future__ import annotations

from pathlib import Path

import pytest

from autoevolve import __version__
from autoevolve.cli._data import load_snapshot, why_ended
from autoevolve.cli.report import report
from tests.fixtures.viz.make_fixture import build_status_fixture


@pytest.mark.parametrize("status", ["target_hit", "budget_exhausted", "plateau", "infeasible"])
def test_report_explains_every_terminal_state(tmp_path: Path, status: str) -> None:
    home = tmp_path / status
    run_id = build_status_fixture(home / "autoevolve.db", status)
    output = tmp_path / f"{status}.md"
    snapshot = load_snapshot(home, run_id)

    result = report(home, run_id, output)
    rendered = result.read_text(encoding="utf-8")

    assert "## Locked contract" in rendered
    assert run_id in rendered
    assert why_ended(snapshot) in rendered
    assert f"autoevolve {__version__}" in rendered
    assert f"recorded seed `{snapshot.run.seed}`" in rendered

