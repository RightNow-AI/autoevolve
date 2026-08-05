"""Batcher odd-even mergesort seed for the sorting-network campaign."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def build(channels: int, deadline: float | None = None) -> list[tuple[int, int]]:
    """Construct Batcher odd-even mergesort with inert high-channel padding."""

    del deadline
    if channels < 2:
        raise ValueError("channels must be at least 2")

    padded_channels = 1
    while padded_channels < channels:
        padded_channels *= 2

    network: list[tuple[int, int]] = []

    def add(left: int, right: int) -> None:
        if right < channels:
            network.append((left, right))

    def merge(start: int, count: int, stride: int) -> None:
        next_stride = stride * 2
        if next_stride < count:
            merge(start, count, next_stride)
            merge(start + stride, count, next_stride)
            for left in range(start + stride, start + count - stride, next_stride):
                add(left, left + stride)
        else:
            add(start, start + stride)

    def sort_range(start: int, count: int) -> None:
        if count <= 1:
            return
        half = count // 2
        sort_range(start, half)
        sort_range(start + half, half)
        merge(start, count, 1)

    sort_range(0, padded_channels)
    return network
# EVOLVE-BLOCK-END
