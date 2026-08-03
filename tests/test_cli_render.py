from __future__ import annotations

from pathlib import Path

from PIL import Image

from autoevolve.cli.render import render_all
from tests.fixtures.viz.make_fixture import build_fixture


def test_render_all_writes_deterministic_public_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    out_dir = tmp_path / "artifacts"
    run_id = build_fixture(home / "autoevolve.db")
    monkeypatch.setattr("autoevolve.cli.render.shutil.which", lambda _: None)

    result = render_all(home, run_id, out_dir)

    for key in ("gif", "poster_svg", "poster_png", "dashboard"):
        path = result[key]
        assert path is not None
        assert path.is_file()
        assert path.stat().st_size > 1_000
    with Image.open(result["gif"]) as image:
        assert 3 < image.n_frames < 121
    assert result["mp4"] is None
    first_poster = result["poster_png"].read_bytes()

    second = render_all(home, run_id, out_dir)
    assert second["poster_png"].read_bytes() == first_poster


def test_live_render_also_writes_latest_png(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    out_dir = tmp_path / "artifacts"
    run_id = build_fixture(home / "autoevolve.db")
    monkeypatch.setattr("autoevolve.cli.render.shutil.which", lambda _: None)

    result = render_all(home, run_id, out_dir, live=True)

    assert result["latest_png"] == out_dir / "latest.png"
    assert result["latest_png"].stat().st_size > 1_000

