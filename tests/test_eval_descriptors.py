"""Program-structure descriptors must separate shapes, not scores."""

from __future__ import annotations

from pathlib import Path

from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, mutable_source, source_metrics

TERSE = """\
import math
# EVOLVE-BLOCK-START
def solve(n):
    return math.isqrt(n)
# EVOLVE-BLOCK-END
"""

ELABORATE = """\
import math
# EVOLVE-BLOCK-START
def solve(n):
    best = None
    for candidate in range(n):
        value = math.sqrt(candidate)
        scaled = round(value)
        joined = sorted([scaled, candidate])
        if best is None or sum(joined) < best:
            best = sum(joined)
    return int(best)
# EVOLVE-BLOCK-END
"""


def test_frozen_code_is_excluded_from_the_measurement() -> None:
    """Frozen code is identical in every candidate and would only add a constant."""

    assert "import math" not in mutable_source(TERSE)
    assert "def solve" in mutable_source(TERSE)


def test_a_file_without_markers_is_entirely_mutable() -> None:
    assert mutable_source("x = 1\n") == "x = 1\n"


def test_two_shapes_land_in_different_cells(tmp_path: Path) -> None:
    terse_dir = tmp_path / "terse"
    terse_dir.mkdir()
    (terse_dir / "main.py").write_text(TERSE, encoding="utf-8")
    elaborate_dir = tmp_path / "elaborate"
    elaborate_dir.mkdir()
    (elaborate_dir / "main.py").write_text(ELABORATE, encoding="utf-8")

    terse = source_metrics(terse_dir, "main.py")
    elaborate = source_metrics(elaborate_dir, "main.py")

    assert elaborate["mutable_lines"] > terse["mutable_lines"]
    assert elaborate["call_diversity"] > terse["call_diversity"]


def test_broken_syntax_still_yields_a_descriptor(tmp_path: Path) -> None:
    """A candidate that cannot parse fails its own gate; it need not crash this."""

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "main.py").write_text("def solve(:\n", encoding="utf-8")

    metrics = source_metrics(broken, "main.py")

    assert metrics["mutable_lines"] == 1.0
    assert metrics["call_diversity"] == 0.0


def test_a_missing_entry_file_is_not_an_error(tmp_path: Path) -> None:
    assert source_metrics(tmp_path, "absent.py") == {
        "mutable_lines": 0.0,
        "call_diversity": 0.0,
    }


def test_every_declared_descriptor_names_a_metric_that_is_produced(tmp_path: Path) -> None:
    entry = tmp_path / "main.py"
    entry.write_text(TERSE, encoding="utf-8")

    produced = source_metrics(tmp_path, "main.py")

    for descriptor in SOURCE_DESCRIPTORS:
        assert descriptor["metric"] in produced
        assert descriptor["bins"] >= 1
        assert descriptor["hi"] > descriptor["lo"]
