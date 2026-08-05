"""Committed mutant that returns a shape-correct but numerically wrong result."""

from __future__ import annotations

from typing import Any

import numpy as np

BLOCK_M = 16
BLOCK_N = 16
KERNEL_LAUNCHES = 1


def run(
    a: Any,
    b: Any,
    bias: Any | None,
    activation: str,
    *,
    real: bool = False,
    deadline: float | None = None,
) -> np.ndarray:
    """Return zeros so the float64 parity gate names the failure."""

    del bias, activation, real, deadline
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    shape = tuple(a_array.shape[:-1]) + (int(b_array.shape[-1]),)
    return np.zeros(shape, dtype=np.float32)
