"""Vector-add-and-scale seed with a lazy GPU implementation.

Candidate contract:
run(x, y, alpha, real=False) returns x + alpha * y.
  real=False returns a NumPy array on the CPU.
  real=True MUST return a CUDA tensor. Returning a CPU result in real mode
  is rejected by the evaluator, because throughput measured on the CPU must
  never be reported as a GPU number.
mock_schedule(n) returns a dictionary containing a numeric score.

Any backend works in real mode: Triton, plain torch, CuPy. Triton ships no
Windows wheel, so the seed uses torch when Triton is unavailable.
"""

from __future__ import annotations

import numpy as np


def ref(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Return the trusted CPU reference without importing GPU packages."""
    return np.asarray(x, dtype=np.float32) + alpha * np.asarray(y, dtype=np.float32)


# EVOLVE-BLOCK-START
BLOCK = 256
num_warps = 4


def mock_schedule(n: int) -> dict[str, float]:
    """Describe launch utilization for the explicitly non-claiming CPU mock."""
    blocks = max(1, (n + BLOCK - 1) // BLOCK)
    lane_utilization = n / (blocks * BLOCK)
    warp_balance = min(num_warps, 4) / max(num_warps, 4)
    return {"score": lane_utilization * warp_balance}


def run(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    real: bool = False,
) -> np.ndarray:
    """Run the NumPy path or lazily compile and launch the Triton kernel."""
    if not real:
        return np.asarray(x, dtype=np.float32) + alpha * np.asarray(y, dtype=np.float32)

    import torch

    device = torch.device("cuda")
    # In real mode the evaluator may hand over device-resident tensors so the
    # timing excludes host transfer. Accept either form.
    x_array = x if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
    y_array = y if torch.is_tensor(y) else np.asarray(y, dtype=np.float32)
    try:
        import triton
        import triton.language as tl
    except ImportError:
        # No Triton on this platform. Measure a real device kernel through
        # torch instead of silently computing on the CPU, which the evaluator
        # rejects in real mode.
        x_tensor = torch.as_tensor(x_array, device=device)
        y_tensor = torch.as_tensor(y_array, device=device)
        return torch.add(x_tensor, y_tensor, alpha=alpha)

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
        tl.store(output_ptr + offsets, x_values + scale * y_values, mask=mask)

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
    return output
# EVOLVE-BLOCK-END
