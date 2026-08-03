"""Plain gradient descent seed for the optimizer-rule proxy task."""

from __future__ import annotations

import numpy as np

OptimizerState = dict[str, np.ndarray | float]

# Candidate contract: preserve both signatures. Evolution may change only the implementations
# inside the EVOLVE-BLOCK markers. Each update must return a new parameter and plain state dict.

# EVOLVE-BLOCK-START
def init_state(shape: tuple[int, ...]) -> OptimizerState:
    """Return empty state because plain SGD has no parameter history."""

    del shape
    return {}


def update(
    param: np.ndarray,
    grad: np.ndarray,
    state: OptimizerState,
    step: int,
) -> tuple[np.ndarray, OptimizerState]:
    """Apply one fixed learning-rate SGD update."""

    del state, step
    learning_rate = 0.05
    new_param = np.array(param - learning_rate * grad, dtype=param.dtype, copy=True)
    return new_param, {}
# EVOLVE-BLOCK-END
