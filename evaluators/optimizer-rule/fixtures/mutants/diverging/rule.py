"""Scripted optimizer mutant whose enormous learning rate diverges."""

from __future__ import annotations

import numpy as np

OptimizerState = dict[str, np.ndarray | float]


# EVOLVE-BLOCK-START
def init_state(shape: tuple[int, ...]) -> OptimizerState:
    del shape
    return {}


def update(
    param: np.ndarray,
    grad: np.ndarray,
    state: OptimizerState,
    step: int,
) -> tuple[np.ndarray, OptimizerState]:
    del state, step
    with np.errstate(over="ignore", invalid="ignore"):
        new_param = np.array(param - 1e308 * grad, dtype=param.dtype, copy=True)
    return new_param, {}
# EVOLVE-BLOCK-END
