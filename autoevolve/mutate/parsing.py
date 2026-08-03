"""Pure parsers for mutation model response formats."""

from __future__ import annotations

import re
from collections.abc import Iterable

SearchReplaceBlock = tuple[str, str, str]

_SEARCH_HEADER = re.compile(r"^<<<<<<< SEARCH[ \t]+([^\r\n]+?)\r?\n?$")
_FILE_HEADER = re.compile(r"^### FILE:[ \t]+([^\r\n]+?)[ \t]*\r?\n?$")
_OPEN_FENCE = re.compile(r"^(`{3,})([^`]*)\r?\n?$")


def parse_search_replace(text: str) -> list[SearchReplaceBlock]:
    """Parse exact SEARCH/REPLACE blocks while preserving block contents."""

    lines = text.splitlines(keepends=True)
    blocks: list[SearchReplaceBlock] = []
    index = 0
    while index < len(lines):
        header = _SEARCH_HEADER.fullmatch(lines[index])
        if header is None:
            index += 1
            continue

        separator = _find_exact_line(lines, index + 1, "=======")
        if separator is None:
            index += 1
            continue
        footer = _find_exact_line(lines, separator + 1, ">>>>>>> REPLACE")
        if footer is None:
            index += 1
            continue

        path = header.group(1).strip()
        search = "".join(lines[index + 1 : separator])
        replace = "".join(lines[separator + 1 : footer])
        blocks.append((path, search, replace))
        index = footer + 1
    return blocks


def _find_exact_line(lines: list[str], start: int, marker: str) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].rstrip("\r\n") == marker:
            return index
    return None


def apply_search_replace(
    files: dict[str, str], blocks: Iterable[SearchReplaceBlock]
) -> tuple[dict[str, str], int, int]:
    """Apply each exact search once and count successful and failed blocks."""

    result = dict(files)
    applied = 0
    failed = 0
    for path, search, replace in blocks:
        content = result.get(path)
        if content is None or not search or search not in content:
            failed += 1
            continue
        result[path] = content.replace(search, replace, 1)
        applied += 1
    return result, applied, failed


def parse_file_blocks(text: str) -> dict[str, str]:
    """Parse path-labeled fenced files, including outer fences over nested fences."""

    lines = text.splitlines(keepends=True)
    files: dict[str, str] = {}
    index = 0
    while index < len(lines):
        header = _FILE_HEADER.fullmatch(lines[index])
        if header is None or index + 1 >= len(lines):
            index += 1
            continue
        opening = _OPEN_FENCE.fullmatch(lines[index + 1])
        if opening is None:
            index += 1
            continue

        fence_length = len(opening.group(1))
        closing = _find_closing_fence(lines, index + 2, fence_length)
        if closing is None:
            index += 1
            continue
        files[header.group(1).strip()] = "".join(lines[index + 2 : closing])
        index = closing + 1
    return files


def _find_closing_fence(lines: list[str], start: int, minimum_length: int) -> int | None:
    for index in range(start, len(lines)):
        candidate = lines[index].rstrip("\r\n").strip()
        if candidate and set(candidate) == {"`"} and len(candidate) >= minimum_length:
            return index
    return None
