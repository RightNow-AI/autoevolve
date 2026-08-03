"""Validation for files whose mutable regions use EVOLVE-BLOCK markers."""

from __future__ import annotations

from dataclasses import dataclass

START = "EVOLVE-BLOCK-START"
END = "EVOLVE-BLOCK-END"


@dataclass(frozen=True)
class BlockCheck:
    """Result of checking a candidate tree against its parent."""

    valid: bool
    reason: str | None = None


def _frozen_lines(text: str) -> tuple[str, ...] | None:
    frozen: list[str] = []
    inside = False
    for line in text.splitlines(keepends=True):
        has_start = START in line
        has_end = END in line
        if has_start and has_end:
            return None
        if has_start:
            if inside:
                return None
            frozen.append(line)
            inside = True
        elif has_end:
            if not inside:
                return None
            frozen.append(line)
            inside = False
        elif not inside:
            frozen.append(line)
    if inside:
        return None
    return tuple(frozen)


def balanced(text: str) -> bool:
    """Return whether marker lines form non-nested balanced regions."""

    return _frozen_lines(text) is not None


def frozen_equal(parent_text: str, child_text: str) -> bool:
    """Compare all text outside balanced mutable regions, including markers."""

    parent_frozen = _frozen_lines(parent_text)
    child_frozen = _frozen_lines(child_text)
    return (
        parent_frozen is not None
        and child_frozen is not None
        and parent_frozen == child_frozen
    )


def validate_files(parent_files: dict[str, str], child_files: dict[str, str]) -> BlockCheck:
    """Enforce frozen regions while allowing ordinary files and new files to change."""

    for path, child_text in child_files.items():
        if not balanced(child_text):
            return BlockCheck(False, f"unbalanced evolve-block markers in {path}")

    for path, parent_text in parent_files.items():
        has_markers = START in parent_text or END in parent_text
        if not has_markers:
            continue
        if not balanced(parent_text):
            return BlockCheck(False, f"parent has unbalanced evolve-block markers in {path}")
        if path not in child_files:
            return BlockCheck(False, f"cannot delete evolve-block file {path}")
        if not frozen_equal(parent_text, child_files[path]):
            return BlockCheck(False, f"frozen text changed in {path}")
    return BlockCheck(True)
