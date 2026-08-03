"""Gate-failing bin-packing mutant used by deterministic tests."""

from __future__ import annotations


def pack(items: list[int], capacity: int) -> list[list[int]]:
    del capacity
    return [[index] for index in range(len(items))] + [[0]]
