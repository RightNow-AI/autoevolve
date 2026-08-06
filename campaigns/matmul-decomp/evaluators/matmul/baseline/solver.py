"""Seeded CP-ALS search with a generated schoolbook fallback."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np


# EVOLVE-BLOCK-START
def _target_tensor(problem: Mapping[str, object], dtype: object) -> np.ndarray:
    m = int(problem["m"])
    k = int(problem["k"])
    n = int(problem["n"])
    target = np.zeros((m * k, k * n, m * n), dtype=dtype)
    for i in range(m):
        for j in range(k):
            for ell in range(n):
                target[i * k + j, j * n + ell, i * n + ell] = 1
    return target


def _schoolbook(problem: Mapping[str, object]) -> dict[str, np.ndarray]:
    m = int(problem["m"])
    k = int(problem["k"])
    n = int(problem["n"])
    dtype = np.complex128 if problem["field"] == "complex" else np.float64
    rank = m * k * n
    u = np.zeros((rank, m * k), dtype=dtype)
    v = np.zeros((rank, k * n), dtype=dtype)
    w = np.zeros((rank, m * n), dtype=dtype)
    row = 0
    for i in range(m):
        for j in range(k):
            for ell in range(n):
                u[row, i * k + j] = 1
                v[row, j * n + ell] = 1
                w[row, i * n + ell] = 1
                row += 1
    return {"U": u, "V": v, "W": w}


def _khatri_rao(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ir,jr->ijr", left, right).reshape(
        left.shape[0] * right.shape[0],
        left.shape[1],
    )


def _normalize_columns(
    left: np.ndarray,
    right: np.ndarray,
    output: np.ndarray,
) -> None:
    for column in range(left.shape[1]):
        left_norm = float(np.linalg.norm(left[:, column]))
        right_norm = float(np.linalg.norm(right[:, column]))
        if left_norm > 0.0:
            left[:, column] /= left_norm
            output[:, column] *= left_norm
        if right_norm > 0.0:
            right[:, column] /= right_norm
            output[:, column] *= right_norm


def _als_attempt(
    target: np.ndarray,
    rank: int,
    rng: np.random.Generator,
    complex_field: bool,
    stop_at: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimensions = target.shape

    def random_factor(rows: int) -> np.ndarray:
        factor = rng.standard_normal((rows, rank))
        if complex_field:
            factor = factor + 1j * rng.standard_normal((rows, rank))
        return factor

    left = random_factor(dimensions[0])
    right = random_factor(dimensions[1])
    output = random_factor(dimensions[2])
    for _ in range(12):
        if time.monotonic() >= stop_at:
            break
        design = _khatri_rao(right, output)
        left = np.linalg.lstsq(design, target.reshape(dimensions[0], -1).T, rcond=None)[
            0
        ].T
        if time.monotonic() >= stop_at:
            break
        design = _khatri_rao(left, output)
        unfolded = np.transpose(target, (1, 0, 2)).reshape(dimensions[1], -1)
        right = np.linalg.lstsq(design, unfolded.T, rcond=None)[0].T
        if time.monotonic() >= stop_at:
            break
        design = _khatri_rao(left, right)
        unfolded = np.transpose(target, (2, 0, 1)).reshape(dimensions[2], -1)
        output = np.linalg.lstsq(design, unfolded.T, rcond=None)[0].T
        _normalize_columns(left, right, output)
    return left.T, right.T, output.T


def _project_exact(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    problem: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerators = tuple(int(value) for value in problem["allowed_numerators"])
    denominator = int(problem["coefficient_denominator"])
    real_values = np.asarray([value / denominator for value in numerators])
    if problem["field"] == "complex":
        grid = np.asarray(
            [real + 1j * imag for real in real_values for imag in real_values],
            dtype=np.complex128,
        )
    else:
        grid = real_values.astype(np.float64)

    projected: list[np.ndarray] = []
    for matrix in matrices:
        distances = np.abs(matrix[..., None] - grid)
        projected.append(grid[np.argmin(distances, axis=-1)])
    return projected[0], projected[1], projected[2]


def _full_identity(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: np.ndarray,
    numeric: bool,
) -> bool:
    u, v, w = matrices
    reconstructed = np.einsum("ra,rb,rc->abc", u, v, w, optimize=False)
    if not numeric:
        return bool(np.array_equal(reconstructed, target))
    magnitude = np.einsum("ra,rb,rc->abc", np.abs(u), np.abs(v), np.abs(w))
    steps = max(1, 3 * u.shape[0])
    epsilon = np.finfo(np.float64).eps
    gamma = steps * epsilon / (1.0 - steps * epsilon)
    scale = np.maximum(magnitude, np.abs(reconstructed))
    scale = np.maximum(scale, np.abs(target))
    return bool(np.all(np.abs(reconstructed - target) <= gamma * scale))


def solve(
    problem: Mapping[str, object],
    deadline: float | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Search at target rank, then return the generated valid fallback."""

    complex_field = problem["field"] == "complex"
    dtype = np.complex128 if complex_field else np.float64
    target = _target_tensor(problem, dtype)
    now = time.monotonic()
    hard_stop = deadline if deadline is not None else now + 2.0
    search_stop = min(hard_stop - 0.25, now + 2.0)
    if search_stop <= now:
        return _schoolbook(problem)

    rank = int(problem["target_rank"])
    rng = np.random.default_rng(int(problem["seed"] if seed is None else seed))
    numeric = problem["coefficient_mode"] == "numeric"
    attempts = 0
    while attempts < 4 and time.monotonic() < search_stop:
        matrices = _als_attempt(target, rank, rng, complex_field, search_stop)
        if not numeric:
            matrices = _project_exact(matrices, problem)
        if _full_identity(matrices, target, numeric):
            return {"U": matrices[0], "V": matrices[1], "W": matrices[2]}
        attempts += 1
    return _schoolbook(problem)
# EVOLVE-BLOCK-END
