from __future__ import annotations

from pathlib import Path

from autoevolve.cli._data import load_snapshot, why_ended
from autoevolve.cli.dashboard import write_dashboard
from tests.fixtures.viz.make_fixture import build_fixture


def test_dashboard_is_self_contained_and_complete(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = build_fixture(home / "autoevolve.db")
    out_path = tmp_path / "dashboard.html"
    snapshot = load_snapshot(home, run_id)

    write_dashboard(home, run_id, out_path)
    rendered = out_path.read_text(encoding="utf-8")

    assert "http://" not in rendered
    assert "https://" not in rendered
    assert run_id in rendered
    assert why_ended(snapshot) in rendered
    assert "Operator and bandit stats" in rendered
    assert rendered.count("<svg ") == 2
    assert "data-tooltip" in rendered

