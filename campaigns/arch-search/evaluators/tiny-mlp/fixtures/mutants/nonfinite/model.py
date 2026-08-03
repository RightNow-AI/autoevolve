"""Gate-failing architecture mutant used by deterministic tests."""

from __future__ import annotations

import numpy as np

INPUT_DIM = 2
HIDDEN_DIM = 8
OUTPUT_DIM = 1
INIT_SCALE = 0.25


def activation(values: np.ndarray) -> np.ndarray:
    return np.full_like(values, np.nan)


def activation_grad(values: np.ndarray) -> np.ndarray:
    return np.full_like(values, np.nan)

