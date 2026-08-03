"""Scripted mutant that adds one to the requested scale."""

from __future__ import annotations

import numpy as np


def ref(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) + alpha * np.asarray(y, dtype=np.float32)


# EVOLVE-BLOCK-START
BLOCK = 256
num_warps = 4


def mock_schedule(n: int) -> dict[str, float]:
    blocks = max(1, (n + BLOCK - 1) // BLOCK)
    return {"score": n / (blocks * BLOCK)}


def run(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    real: bool = False,
) -> np.ndarray:
    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    if not real:
        return x_array + (alpha + 1.0) * y_array

    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError:
        return x_array + (alpha + 1.0) * y_array

    @triton.jit
    def vector_add_scale_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        scale,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_values = tl.load(x_ptr + offsets, mask=mask)
        y_values = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x_values + (scale + 1.0) * y_values, mask=mask)

    device = torch.device("cuda")
    x_tensor = torch.as_tensor(x_array, device=device)
    y_tensor = torch.as_tensor(y_array, device=device)
    output = torch.empty_like(x_tensor)
    grid = (triton.cdiv(x_tensor.numel(), BLOCK),)
    vector_add_scale_kernel[grid](
        x_tensor,
        y_tensor,
        output,
        alpha,
        x_tensor.numel(),
        BLOCK_SIZE=BLOCK,
        num_warps=num_warps,
    )
    return output.cpu().numpy()
# EVOLVE-BLOCK-END
