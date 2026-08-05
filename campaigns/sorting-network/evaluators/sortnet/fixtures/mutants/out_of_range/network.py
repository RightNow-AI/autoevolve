"""Invalid sorting network with one comparator outside the channel range."""

from __future__ import annotations


def build(channels: int, deadline: float | None = None) -> list[tuple[int, int]]:
    del deadline
    return [(0, channels)]
