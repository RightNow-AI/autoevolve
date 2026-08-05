"""Remote CUDA selftest for the exact Ramsey annealer.

The delta test compares torch bitset results with an independent plain Python
enumeration of every clique containing the flipped edge. It covers K5 and K4
changes on changing random colorings for n from 8 through 14. The search test
then requires a host-verified zero-K5 result at the known easy size n = 30.

This file is intended to run inside the Modal GPU image. It is deliberately not
part of local CPU CI.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from itertools import combinations
from pathlib import Path

import torch
from gpu_anneal import (
    OBJECTIVE_K5,
    blue_rows,
    build_geometry,
    compute_flip_deltas,
    search,
)


def _random_coloring(n: int, rng: random.Random) -> list[int]:
    """Create one symmetric random coloring as red adjacency rows."""

    red = [0] * n
    for left in range(n):
        for right in range(left + 1, n):
            if rng.getrandbits(1):
                red[left] |= 1 << right
                red[right] |= 1 << left
    return red


def _is_monochromatic(red: list[int], vertices: tuple[int, ...]) -> bool:
    """Judge one clique candidate directly from all of its pairs."""

    colors = [bool(red[left] >> right & 1) for left, right in combinations(vertices, 2)]
    return all(colors) or not any(colors)


def _brute_flip_delta(
    n: int,
    red: list[int],
    u: int,
    v: int,
    clique_size: int,
) -> int:
    """Enumerate every affected clique before and after one edge flip."""

    before = 0
    after = 0
    other_vertices = [vertex for vertex in range(n) if vertex not in (u, v)]
    for remainder in combinations(other_vertices, clique_size - 2):
        vertices = tuple(sorted((u, v, *remainder)))
        before += int(_is_monochromatic(red, vertices))
        red[u] ^= 1 << v
        red[v] ^= 1 << u
        after += int(_is_monochromatic(red, vertices))
        red[u] ^= 1 << v
        red[v] ^= 1 << u
    return after - before


def validate_deltas(
    *,
    device: str,
    states: int,
    flips_per_state: int,
    seed: int,
) -> dict[str, int]:
    """Assert exact GPU agreement over many random states and flips."""

    if states <= 0 or flips_per_state <= 0:
        raise ValueError("states and flips_per_state must be positive")
    rng = random.Random(seed)
    comparisons = 0
    for n in range(8, 15):
        geometry = build_geometry(n, device)
        colorings = [_random_coloring(n, rng) for _ in range(states)]
        for flip_index in range(flips_per_state):
            endpoints = [tuple(sorted(rng.sample(range(n), 2))) for _ in range(states)]
            u_host = [left for left, _ in endpoints]
            v_host = [right for _, right in endpoints]
            expected_k5 = [
                _brute_flip_delta(n, red, u, v, 5)
                for red, u, v in zip(colorings, u_host, v_host, strict=True)
            ]
            expected_k4 = [
                _brute_flip_delta(n, red, u, v, 4)
                for red, u, v in zip(colorings, u_host, v_host, strict=True)
            ]

            red_tensor = torch.tensor(colorings, dtype=torch.int64, device=device)
            blue_tensor = blue_rows(red_tensor, geometry)
            u_tensor = torch.tensor(u_host, dtype=torch.int64, device=device)
            v_tensor = torch.tensor(v_host, dtype=torch.int64, device=device)
            observed = compute_flip_deltas(
                red_tensor,
                blue_tensor,
                u_tensor,
                v_tensor,
                geometry,
                include_k4=True,
            )
            observed_k5 = [int(value) for value in observed.k5.cpu().tolist()]
            if observed.k4 is None:
                raise AssertionError("K4 delta was not returned")
            observed_k4 = [int(value) for value in observed.k4.cpu().tolist()]
            if observed_k5 != expected_k5:
                raise AssertionError(
                    f"K5 delta mismatch at n={n}, flip batch={flip_index}: "
                    f"expected={expected_k5}, observed={observed_k5}"
                )
            if observed_k4 != expected_k4:
                raise AssertionError(
                    f"K4 delta mismatch at n={n}, flip batch={flip_index}: "
                    f"expected={expected_k4}, observed={observed_k4}"
                )

            for red, u, v in zip(colorings, u_host, v_host, strict=True):
                red[u] ^= 1 << v
                red[v] ^= 1 << u
            comparisons += states
        print(
            f"delta exactness n={n}: {states * flips_per_state} changing-state flips",
            flush=True,
        )
    return {"delta_comparisons": comparisons, "n_min": 8, "n_max": 14}


def validate_easy_search(
    *,
    device: str,
    batch_size: int,
    seconds: float,
    max_steps: int,
    seed: int,
) -> dict[str, object]:
    """Require the GPU search to produce and host-verify an n = 30 certificate."""

    with tempfile.TemporaryDirectory(prefix="autoevolve-gpu-selftest-") as temporary:
        result = search(
            n=30,
            objective=OBJECTIVE_K5,
            batch_size=batch_size,
            seconds=seconds,
            max_steps=max_steps,
            seed=seed,
            device=device,
            output_dir=Path(temporary),
            sync_every=5,
            report_every_seconds=30.0,
        )
        if result.verified_monochromatic_k5s != 0:
            raise AssertionError(
                "easy search did not reach zero monochromatic K5s at n=30: "
                f"best={result.verified_monochromatic_k5s}"
            )
        if result.certificate_path is None:
            raise AssertionError("easy search reached zero but emitted no verified certificate")
        return result.to_dict()


def main() -> None:
    """Run both exactness gates inside a CUDA container."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta-states", type=int, default=24)
    parser.add_argument("--delta-flips", type=int, default=32)
    parser.add_argument("--delta-seed", type=int, default=20260805)
    parser.add_argument("--easy-batch-size", type=int, default=8192)
    parser.add_argument("--easy-seconds", type=float, default=180.0)
    parser.add_argument("--easy-max-steps", type=int, default=200_000)
    parser.add_argument("--easy-seed", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_anneal_selftest requires CUDA")

    delta_report = validate_deltas(
        device=args.device,
        states=args.delta_states,
        flips_per_state=args.delta_flips,
        seed=args.delta_seed,
    )
    easy_report = validate_easy_search(
        device=args.device,
        batch_size=args.easy_batch_size,
        seconds=args.easy_seconds,
        max_steps=args.easy_max_steps,
        seed=args.easy_seed,
    )
    print(
        json.dumps(
            {"delta": delta_report, "easy_search": easy_report, "status": "passed"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
