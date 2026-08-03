"""EVOLVE-BLOCK frozen-region discipline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoevolve.core.evolve_blocks import balanced, frozen_equal, validate_files


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def test_frozen_equal_allows_changes_inside_one_region(home: Path) -> None:
    parent = "before\n# EVOLVE-BLOCK-START\nold\n# EVOLVE-BLOCK-END\nafter\n"
    child = "before\n# EVOLVE-BLOCK-START\nnew\n# EVOLVE-BLOCK-END\nafter\n"
    assert frozen_equal(parent, child)


def test_frozen_equal_supports_multiple_regions(home: Path) -> None:
    parent = (
        "head\n"
        "# EVOLVE-BLOCK-START\none\n# EVOLVE-BLOCK-END\n"
        "middle\n"
        "# EVOLVE-BLOCK-START\ntwo\n# EVOLVE-BLOCK-END\n"
        "tail\n"
    )
    child = (
        "head\n"
        "# EVOLVE-BLOCK-START\nchanged one\n# EVOLVE-BLOCK-END\n"
        "middle\n"
        "# EVOLVE-BLOCK-START\nchanged two\nextra\n# EVOLVE-BLOCK-END\n"
        "tail\n"
    )
    assert frozen_equal(parent, child)


def test_frozen_equal_rejects_change_outside_regions(home: Path) -> None:
    parent = "before\n# EVOLVE-BLOCK-START\nold\n# EVOLVE-BLOCK-END\nafter\n"
    child = "changed\n# EVOLVE-BLOCK-START\nold\n# EVOLVE-BLOCK-END\nafter\n"
    assert not frozen_equal(parent, child)


@pytest.mark.parametrize(
    "text",
    [
        "# EVOLVE-BLOCK-START\nmissing end\n",
        "# EVOLVE-BLOCK-END\n",
        "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END\n",
        "# EVOLVE-BLOCK-START EVOLVE-BLOCK-END\n",
    ],
)
def test_unbalanced_or_nested_markers_are_rejected(home: Path, text: str) -> None:
    assert not balanced(text)
    assert not frozen_equal(text, text)


def test_validation_allows_unmarked_changes_and_new_files(home: Path) -> None:
    result = validate_files(
        {"mutable.py": "old\n"},
        {"mutable.py": "new\n", "new.py": "created\n"},
    )
    assert result.valid


def test_validation_rejects_marker_file_deletion(home: Path) -> None:
    parent = {"fenced.py": "# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\n"}
    result = validate_files(parent, {})
    assert not result.valid
    assert "cannot delete" in str(result.reason)


def test_validation_rejects_demarker_and_unbalanced_new_file(home: Path) -> None:
    parent = {"fenced.py": "# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\n"}
    assert not validate_files(parent, {"fenced.py": "x\n"}).valid
    result = validate_files(parent, {**parent, "new.py": "# EVOLVE-BLOCK-START\n"})
    assert not result.valid
    assert "unbalanced" in str(result.reason)
