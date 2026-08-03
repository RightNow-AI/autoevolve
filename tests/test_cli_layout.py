from __future__ import annotations

from pathlib import Path

from autoevolve.cli._data import load_snapshot
from autoevolve.cli.render import compute_layout
from tests.fixtures.viz.make_fixture import build_fixture


def test_layout_is_incremental_and_layered(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = build_fixture(home / "autoevolve.db")
    snapshot = load_snapshot(home, run_id)
    programs = [program.as_tuple() for program in snapshot.programs]
    edges = list(snapshot.edges)
    full = compute_layout(programs, edges)

    for size in (1, 4, 9, 17, 28, len(programs)):
        prefix = compute_layout(programs[:size], edges)
        expected = {str(program[0]): full[str(program[0])] for program in programs[:size]}
        assert prefix == expected

    lane_ranges: dict[int, tuple[float, float]] = {}
    for island in range(3):
        x_values = [
            full[program.id][0]
            for program in snapshot.programs
            if program.island == island
        ]
        lane_ranges[island] = (min(x_values), max(x_values))
    assert lane_ranges[0][1] < lane_ranges[1][0]
    assert lane_ranges[1][1] < lane_ranges[2][0]

    for child_id, parent_id, _ in snapshot.edges:
        assert full[child_id][1] > full[parent_id][1]
