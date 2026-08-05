"""Untuned two-launch Triton GEMM seed written from first principles.

The evaluator calls run(a, b, bias, activation, real, deadline). Real inputs are
device-resident float32 CUDA tensors. Mock inputs are NumPy arrays. The real result
must be a float32 CUDA tensor on the same device.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# EVOLVE-BLOCK-START
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
NUM_WARPS = 4
NUM_STAGES = 2
EPILOGUE_BLOCK = 128
KERNEL_LAUNCHES = 2


def run(
    a: Any,
    b: Any,
    bias: Any | None,
    activation: str,
    *,
    real: bool = False,
    deadline: float | None = None,
) -> object:
    """Run a simple tiled GEMM followed by an explicit epilogue kernel."""

    del deadline
    if not real:
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        output = np.matmul(a_array, b_array)
        if bias is not None:
            output = output + np.asarray(bias, dtype=np.float32)
        if activation == "relu":
            output = np.maximum(output, np.float32(0.0))
        return np.asarray(output, dtype=np.float32)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def tiled_gemm_kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        m,
        n,
        k,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_ob,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        program_mn = tl.program_id(axis=0)
        program_batch = tl.program_id(axis=1)
        programs_n = tl.cdiv(n, BLOCK_SIZE_N)
        program_m = program_mn // programs_n
        program_n = program_mn % programs_n
        offsets_m = program_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = program_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offsets_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = (
            a_ptr
            + program_batch * stride_ab
            + offsets_m[:, None] * stride_am
            + offsets_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + program_batch * stride_bb
            + offsets_k[:, None] * stride_bk
            + offsets_n[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for start_k in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
            remaining = k - start_k * BLOCK_SIZE_K
            a_values = tl.load(
                a_ptrs,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < remaining),
                other=0.0,
            )
            b_values = tl.load(
                b_ptrs,
                mask=(offsets_k[:, None] < remaining) & (offsets_n[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(a_values, b_values, input_precision="ieee")
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        output_offsets = (
            program_batch * stride_ob
            + offsets_m[:, None] * stride_om
            + offsets_n[None, :] * stride_on
        )
        output_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
        tl.store(output_ptr + output_offsets, accumulator, mask=output_mask)

    @triton.jit
    def epilogue_kernel(
        input_ptr,
        bias_ptr,
        output_ptr,
        elements,
        width,
        HAS_BIAS: tl.constexpr,
        APPLY_RELU: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        values = tl.load(input_ptr + offsets, mask=mask)
        if HAS_BIAS:
            columns = offsets % width
            values += tl.load(bias_ptr + columns, mask=mask)
        if APPLY_RELU:
            values = tl.maximum(values, 0.0)
        tl.store(output_ptr + offsets, values, mask=mask)

    a_tensor = a
    b_tensor = b
    if not torch.is_tensor(a_tensor) or not torch.is_tensor(b_tensor):
        raise TypeError("real mode requires CUDA tensors")
    if a_tensor.ndim not in {2, 3} or b_tensor.ndim != a_tensor.ndim:
        raise ValueError("a and b must both be rank two or both be rank three")
    batch = 1 if a_tensor.ndim == 2 else int(a_tensor.shape[0])
    m = int(a_tensor.shape[-2])
    k = int(a_tensor.shape[-1])
    n = int(b_tensor.shape[-1])
    output_shape = tuple(a_tensor.shape[:-2]) + (m, n)
    scratch = torch.empty(output_shape, device=a_tensor.device, dtype=torch.float32)
    output = torch.empty_like(scratch)
    stride_ab = 0 if a_tensor.ndim == 2 else a_tensor.stride(0)
    stride_bb = 0 if b_tensor.ndim == 2 else b_tensor.stride(0)
    stride_ob = 0 if scratch.ndim == 2 else scratch.stride(0)
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), batch)
    tiled_gemm_kernel[grid](
        a_tensor,
        b_tensor,
        scratch,
        m,
        n,
        k,
        stride_ab,
        a_tensor.stride(-2),
        a_tensor.stride(-1),
        stride_bb,
        b_tensor.stride(-2),
        b_tensor.stride(-1),
        stride_ob,
        scratch.stride(-2),
        scratch.stride(-1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    bias_pointer = scratch if bias is None else bias
    epilogue_grid = (triton.cdiv(output.numel(), EPILOGUE_BLOCK),)
    epilogue_kernel[epilogue_grid](
        scratch,
        bias_pointer,
        output,
        output.numel(),
        n,
        HAS_BIAS=bias is not None,
        APPLY_RELU=activation == "relu",
        BLOCK_SIZE=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return output
# EVOLVE-BLOCK-END
