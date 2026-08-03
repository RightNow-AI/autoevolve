"""First-fit seed for one-dimensional bin packing."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def pack(items: list[int], capacity: int) -> list[list[int]]:
    """Place item indexes into bins with the first-fit rule."""

    bins: list[list[int]] = []
    loads: list[int] = []
    for index, size in enumerate(items):
        for bin_index, load in enumerate(loads):
            if load + size <= capacity:
                bins[bin_index].append(index)
                loads[bin_index] += size
                break
        else:
            bins.append([index])
            loads.append(size)
    return bins
# EVOLVE-BLOCK-END

