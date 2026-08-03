"""Small NumPy MLP candidate with one mutable activation block."""

from __future__ import annotations

import numpy as np

INPUT_DIM = 2
HIDDEN_DIM = 8
OUTPUT_DIM = 1


# EVOLVE-BLOCK-START
INIT_SCALE = 0.25


def activation(values: np.ndarray) -> np.ndarray:
    """Return the hidden activation."""

    return np.tanh(values)


def activation_grad(values: np.ndarray) -> np.ndarray:
    """Return the hidden activation derivative."""

    activated = np.tanh(values)
    return 1.0 - activated * activated
# EVOLVE-BLOCK-END

