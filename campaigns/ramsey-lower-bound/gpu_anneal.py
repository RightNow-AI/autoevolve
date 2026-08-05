"""Batched exact simulated annealing for the Ramsey R(5,5) frontier.

Each CUDA lane owns one complete coloring as ``n`` signed int64 adjacency rows.
For the intended n = 43 problem every used bit is below the sign bit. A step
chooses one independent random edge per lane, computes the exact change in the
number of monochromatic K5s, and applies the Metropolis rule independently.

The flip delta is local. Recoloring edge uv only changes monochromatic cliques
that contain u and v. The other three vertices of a K5 must form a triangle in
the common neighborhood of u and v in the relevant color. The common
neighborhood itself does not change when uv is recolored, because adjacency
rows have zero diagonal bits. K4 deltas use the same argument with an edge
inside the common neighborhood.

No GPU result is emitted as a certificate until plain Python recounts every
five-set in the selected coloring and agrees with the tracked exact cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import torch

OBJECTIVE_K5: Final = "min_k5"
OBJECTIVE_K4_AT_ZERO_K5: Final = "min_k4_at_zero_k5"
OBJECTIVES: Final = (OBJECTIVE_K5, OBJECTIVE_K4_AT_ZERO_K5)
CPU_REFERENCE_FLIPS_PER_SECOND: Final = 58_000.0


@dataclass(frozen=True)
class GraphGeometry:
    """CUDA constants shared by every independent annealing lane."""

    n: int
    full_mask: int
    vertex_bits: torch.Tensor
    vertex_indices: torch.Tensor
    higher_masks: torch.Tensor
    edge_left: torch.Tensor
    edge_right: torch.Tensor
    popcount8: torch.Tensor


@dataclass(frozen=True)
class FlipDeltas:
    """Exact objective changes for one proposed edge in every lane."""

    k5: torch.Tensor
    k4: torch.Tensor | None


@dataclass(frozen=True)
class SearchResult:
    """Host-verified result and measured throughput from one CUDA run."""

    objective: str
    n: int
    batch_size: int
    seed: int
    steps: int
    aggregate_flips: int
    elapsed_seconds: float
    aggregate_flips_per_second: float
    provided_cpu_reference_flips_per_second: float
    multiple_of_provided_cpu_reference: float
    verified_monochromatic_k5s: int
    verified_monochromatic_k4s: int
    stop_reason: str
    certificate_path: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready report."""

        return asdict(self)


def build_geometry(n: int, device: torch.device | str) -> GraphGeometry:
    """Build exact bit masks and the byte popcount table on one device."""

    if not 5 <= n <= 62:
        raise ValueError("n must be between 5 and 62 so every row stays nonnegative int64")
    target = torch.device(device)
    full_mask = (1 << n) - 1
    vertex_indices = torch.arange(n, dtype=torch.int64, device=target)
    vertex_bits = torch.tensor(
        [1 << vertex for vertex in range(n)],
        dtype=torch.int64,
        device=target,
    )
    higher_masks = torch.tensor(
        [full_mask ^ ((1 << (vertex + 1)) - 1) for vertex in range(n)],
        dtype=torch.int64,
        device=target,
    )
    edges = list(combinations(range(n), 2))
    edge_left = torch.tensor(
        [left for left, _ in edges],
        dtype=torch.int64,
        device=target,
    )
    edge_right = torch.tensor(
        [right for _, right in edges],
        dtype=torch.int64,
        device=target,
    )
    # A 256-entry byte lookup is used instead of SWAR. It is slower than a fused
    # hardware popcount, but its exact behavior on signed torch.int64 values is
    # transparent and does not depend on overflow or arithmetic right shifts.
    popcount8 = torch.tensor(
        [value.bit_count() for value in range(256)],
        dtype=torch.int64,
        device=target,
    )
    return GraphGeometry(
        n=n,
        full_mask=full_mask,
        vertex_bits=vertex_bits,
        vertex_indices=vertex_indices,
        higher_masks=higher_masks,
        edge_left=edge_left,
        edge_right=edge_right,
        popcount8=popcount8,
    )


def blue_rows(red: torch.Tensor, geometry: GraphGeometry) -> torch.Tensor:
    """Return the exact complementary coloring with every diagonal bit clear."""

    diagonal_clear = torch.bitwise_xor(geometry.vertex_bits, geometry.full_mask)
    return torch.bitwise_and(
        torch.bitwise_xor(red, geometry.full_mask),
        diagonal_clear.unsqueeze(0),
    )


def _popcount_i64(values: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Count set bits in nonnegative int64 tensors by exact byte lookup."""

    counts = torch.zeros_like(values)
    for shift in range(0, 64, 8):
        byte_values = torch.bitwise_and(torch.bitwise_right_shift(values, shift), 0xFF)
        counts = counts + table[byte_values]
    return counts


def _edges_in_masks(
    rows: torch.Tensor,
    masks: torch.Tensor,
    geometry: GraphGeometry,
) -> torch.Tensor:
    """Count same-color edges induced by one vertex mask per lane."""

    restricted = torch.bitwise_and(rows, masks.unsqueeze(1))
    restricted = torch.bitwise_and(restricted, geometry.higher_masks.unsqueeze(0))
    vertex_present = torch.bitwise_and(
        torch.bitwise_right_shift(
            masks.unsqueeze(1),
            geometry.vertex_indices.unsqueeze(0),
        ),
        1,
    )
    return (_popcount_i64(restricted, geometry.popcount8) * vertex_present).sum(dim=1)


def _triangles_in_masks(
    rows: torch.Tensor,
    masks: torch.Tensor,
    geometry: GraphGeometry,
    edge_chunk: int,
) -> torch.Tensor:
    """Count triangles induced by one mask per lane, once by vertex order."""

    if edge_chunk <= 0:
        raise ValueError("edge_chunk must be positive")
    totals = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
    edge_count = geometry.edge_left.numel()
    for start in range(0, edge_count, edge_chunk):
        stop = min(start + edge_chunk, edge_count)
        left = geometry.edge_left[start:stop]
        right = geometry.edge_right[start:stop]
        left_rows = rows[:, left]
        right_rows = rows[:, right]
        third_vertices = torch.bitwise_and(left_rows, right_rows)
        third_vertices = torch.bitwise_and(third_vertices, masks.unsqueeze(1))
        third_vertices = torch.bitwise_and(
            third_vertices,
            geometry.higher_masks[right].unsqueeze(0),
        )
        edge_present = torch.bitwise_and(
            torch.bitwise_right_shift(left_rows, right.unsqueeze(0)),
            1,
        )
        left_present = torch.bitwise_and(
            torch.bitwise_right_shift(masks.unsqueeze(1), left.unsqueeze(0)),
            1,
        )
        right_present = torch.bitwise_and(
            torch.bitwise_right_shift(masks.unsqueeze(1), right.unsqueeze(0)),
            1,
        )
        triangle_counts = _popcount_i64(third_vertices, geometry.popcount8)
        totals = totals + (
            triangle_counts * edge_present * left_present * right_present
        ).sum(dim=1)
    return totals


def compute_flip_deltas(
    red: torch.Tensor,
    blue: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    geometry: GraphGeometry,
    *,
    include_k4: bool,
    edge_chunk: int = 128,
) -> FlipDeltas:
    """Compute exact K5 and optional K4 deltas for one flip per lane."""

    batch_size, n = red.shape
    if blue.shape != red.shape or n != geometry.n:
        raise ValueError("red and blue rows must match the geometry")
    if u.shape != (batch_size,) or v.shape != (batch_size,):
        raise ValueError("u and v must contain one endpoint per lane")

    lane = torch.arange(batch_size, device=red.device)
    red_u = red[lane, u]
    red_v = red[lane, v]
    blue_u = blue[lane, u]
    blue_v = blue[lane, v]
    bit_u = geometry.vertex_bits[u]
    bit_v = geometry.vertex_bits[v]
    other_vertices = torch.bitwise_xor(bit_u, geometry.full_mask)
    other_vertices = torch.bitwise_xor(other_vertices, bit_v)
    red_common = torch.bitwise_and(torch.bitwise_and(red_u, red_v), other_vertices)
    blue_common = torch.bitwise_and(torch.bitwise_and(blue_u, blue_v), other_vertices)

    both_rows = torch.cat((red, blue), dim=0)
    both_common = torch.cat((red_common, blue_common), dim=0)
    through_k5 = _triangles_in_masks(both_rows, both_common, geometry, edge_chunk)
    red_k5, blue_k5 = through_k5.split(batch_size)
    was_red = torch.bitwise_and(red_u, bit_v) != 0
    k5_delta = torch.where(was_red, blue_k5 - red_k5, red_k5 - blue_k5)

    k4_delta = None
    if include_k4:
        through_k4 = _edges_in_masks(both_rows, both_common, geometry)
        red_k4, blue_k4 = through_k4.split(batch_size)
        k4_delta = torch.where(was_red, blue_k4 - red_k4, red_k4 - blue_k4)
    return FlipDeltas(k5=k5_delta, k4=k4_delta)


def _validate_host_rows(n: int, red: list[int]) -> None:
    """Reject malformed host colorings before counting or emission."""

    if len(red) != n:
        raise ValueError(f"expected {n} adjacency rows, got {len(red)}")
    full_mask = (1 << n) - 1
    for vertex, row in enumerate(red):
        if not isinstance(row, int) or isinstance(row, bool):
            raise ValueError("adjacency rows must be plain integers")
        if row < 0 or row & ~full_mask:
            raise ValueError(f"row {vertex} contains a bit outside 0..{n - 1}")
        if row >> vertex & 1:
            raise ValueError(f"row {vertex} contains a self loop")
    for left in range(n):
        for right in range(left + 1, n):
            if bool(red[left] >> right & 1) != bool(red[right] >> left & 1):
                raise ValueError(f"adjacency is not symmetric at edge {left},{right}")


def count_monochromatic_cliques(n: int, red: list[int], clique_size: int) -> int:
    """Recount every clique candidate in plain Python with exact integers."""

    _validate_host_rows(n, red)
    edge_total = clique_size * (clique_size - 1) // 2
    total = 0
    for vertices in combinations(range(n), clique_size):
        red_edges = sum(
            1 for left, right in combinations(vertices, 2) if red[left] >> right & 1
        )
        if red_edges in (0, edge_total):
            total += 1
    return total


def load_adjacency_certificate(path: Path) -> tuple[int, list[int]]:
    """Load the campaign's strict adjacency certificate form."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"form", "n", "red_edges"}:
        raise ValueError(f"{path} is not an exact adjacency certificate")
    if data["form"] != "adjacency":
        raise ValueError(f"{path} uses unsupported form {data['form']!r}")
    n = data["n"]
    if not isinstance(n, int) or isinstance(n, bool) or not 5 <= n <= 62:
        raise ValueError(f"{path} has invalid n")
    edges = data["red_edges"]
    if not isinstance(edges, list):
        raise ValueError(f"{path} red_edges must be a list")

    red = [0] * n
    previous: tuple[int, int] | None = None
    for pair in edges:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{path} contains a malformed edge")
        left, right = pair
        if any(not isinstance(value, int) or isinstance(value, bool) for value in pair):
            raise ValueError(f"{path} contains a noninteger edge")
        if not 0 <= left < right < n:
            raise ValueError(f"{path} contains invalid edge {pair}")
        edge = (left, right)
        if previous is not None and edge <= previous:
            raise ValueError(f"{path} red_edges are not strictly increasing")
        previous = edge
        red[left] |= 1 << right
        red[right] |= 1 << left
    _validate_host_rows(n, red)
    return n, red


def _initial_coloring(n: int, seed: int, start_path: Path | None) -> list[int]:
    """Build one exact seed coloring without embedding any frontier answer."""

    rng = random.Random(seed)
    if start_path is None:
        red = [0] * n
        for left in range(n):
            for right in range(left + 1, n):
                if rng.getrandbits(1):
                    red[left] |= 1 << right
                    red[right] |= 1 << left
        return red

    start_n, red = load_adjacency_certificate(start_path)
    if start_n == n:
        return red
    if start_n != n - 1:
        raise ValueError(f"start certificate has n={start_n}; expected {n} or {n - 1}")

    # A verified (n-1)-vertex certificate is only a seed. Random incident
    # colors for the new vertex are recounted exactly and make no result claim.
    red.append(0)
    new_vertex = n - 1
    for vertex in range(new_vertex):
        if rng.getrandbits(1):
            red[vertex] |= 1 << new_vertex
            red[new_vertex] |= 1 << vertex
    _validate_host_rows(n, red)
    return red


def _write_certificate(output_dir: Path, n: int, red: list[int]) -> Path:
    """Atomically write canonical certificate JSON after exact host verification."""

    edges = [
        [left, right]
        for left in range(n)
        for right in range(left + 1, n)
        if red[left] >> right & 1
    ]
    payload = {"form": "adjacency", "n": n, "red_edges": edges}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"n{n}-{digest}.json"
    temporary = output_dir / f".{target.name}.{os.getpid()}.tmp"
    if target.is_file():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"certificate hash collision at {target}")
        return target
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    return target


def _apply_accepted_flips(
    red: torch.Tensor,
    blue: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    accepted: torch.Tensor,
    geometry: GraphGeometry,
) -> None:
    """Apply one accepted or rejected proposal independently in every lane."""

    lane = torch.arange(red.shape[0], device=red.device)
    accepted_i64 = accepted.to(torch.int64)
    toggle_u = geometry.vertex_bits[v] * accepted_i64
    toggle_v = geometry.vertex_bits[u] * accepted_i64
    old_red_u = red[lane, u]
    old_red_v = red[lane, v]
    old_blue_u = blue[lane, u]
    old_blue_v = blue[lane, v]
    red[lane, u] = torch.bitwise_xor(old_red_u, toggle_u)
    red[lane, v] = torch.bitwise_xor(old_red_v, toggle_v)
    blue[lane, u] = torch.bitwise_xor(old_blue_u, toggle_u)
    blue[lane, v] = torch.bitwise_xor(old_blue_v, toggle_v)


def search(
    *,
    n: int,
    objective: str,
    batch_size: int,
    seconds: float,
    max_steps: int,
    seed: int,
    device: str = "cuda",
    start_path: Path | None = None,
    output_dir: Path | None = None,
    start_temperature: float | None = None,
    end_temperature: float | None = None,
    edge_chunk: int = 128,
    sync_every: int = 10,
    report_every_seconds: float = 30.0,
) -> SearchResult:
    """Run B independent exact annealing chains on one CUDA device."""

    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if batch_size <= 0 or seconds <= 0 or max_steps <= 0:
        raise ValueError("batch_size, seconds, and max_steps must be positive")
    if sync_every <= 0 or report_every_seconds <= 0:
        raise ValueError("sync and report intervals must be positive")
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the annealer requires an available CUDA device")

    initial_red = _initial_coloring(n, seed, start_path)
    initial_k5 = count_monochromatic_cliques(n, initial_red, 5)
    initial_k4 = count_monochromatic_cliques(n, initial_red, 4)
    if objective == OBJECTIVE_K4_AT_ZERO_K5 and initial_k5 != 0:
        raise ValueError(
            "min_k4_at_zero_k5 requires an exactly K5-free n-vertex start certificate"
        )

    geometry = build_geometry(n, target_device)
    red = torch.tensor(initial_red, dtype=torch.int64, device=target_device)
    red = red.unsqueeze(0).repeat(batch_size, 1)
    blue = blue_rows(red, geometry)
    k5_cost = torch.full(
        (batch_size,),
        initial_k5,
        dtype=torch.int64,
        device=target_device,
    )
    k4_cost = torch.full(
        (batch_size,),
        initial_k4,
        dtype=torch.int64,
        device=target_device,
    )
    best_k5 = k5_cost.clone()
    best_k4 = k4_cost.clone()
    best_rows = red.clone()

    if start_temperature is None:
        start_temperature = 3.0 if objective == OBJECTIVE_K5 else 60.0
    if end_temperature is None:
        end_temperature = 0.05 if objective == OBJECTIVE_K5 else 0.5
    if start_temperature <= 0 or end_temperature <= 0:
        raise ValueError("temperatures must be positive")

    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    temperature_scale = torch.exp(
        torch.linspace(
            math.log(0.5),
            math.log(2.0),
            batch_size,
            dtype=torch.float32,
            device=target_device,
        )
    )
    edge_count = geometry.edge_left.numel()
    torch.cuda.synchronize(target_device)
    started = time.perf_counter()
    elapsed = 0.0
    next_report = report_every_seconds
    steps = 0
    stop_reason = "max_steps"

    while steps < max_steps:
        progress = max(steps / max_steps, min(elapsed / seconds, 1.0))
        base_temperature = start_temperature * (
            end_temperature / start_temperature
        ) ** progress
        temperature = temperature_scale * base_temperature

        edge_index = torch.randint(
            edge_count,
            (batch_size,),
            generator=generator,
            device=target_device,
        )
        u = geometry.edge_left[edge_index]
        v = geometry.edge_right[edge_index]
        deltas = compute_flip_deltas(
            red,
            blue,
            u,
            v,
            geometry,
            include_k4=objective == OBJECTIVE_K4_AT_ZERO_K5,
            edge_chunk=edge_chunk,
        )
        candidate_k5 = k5_cost + deltas.k5
        if objective == OBJECTIVE_K5:
            objective_delta = deltas.k5
            candidate_k4 = k4_cost
            feasible = torch.ones(batch_size, dtype=torch.bool, device=target_device)
        else:
            if deltas.k4 is None:
                raise RuntimeError("K4 objective requested without an exact K4 delta")
            objective_delta = deltas.k4
            candidate_k4 = k4_cost + deltas.k4
            feasible = candidate_k5 == 0

        uphill = torch.clamp(objective_delta, min=0).to(torch.float32)
        probability = torch.exp(-uphill / temperature)
        draw = torch.rand(batch_size, generator=generator, device=target_device)
        accepted = feasible & ((objective_delta <= 0) | (draw < probability))
        _apply_accepted_flips(red, blue, u, v, accepted, geometry)
        k5_cost = torch.where(accepted, candidate_k5, k5_cost)
        k4_cost = torch.where(accepted, candidate_k4, k4_cost)

        if objective == OBJECTIVE_K5:
            improved = accepted & (candidate_k5 < best_k5)
            best_k5 = torch.where(improved, candidate_k5, best_k5)
        else:
            improved = accepted & (candidate_k4 < best_k4)
            best_k4 = torch.where(improved, candidate_k4, best_k4)
        best_rows = torch.where(improved.unsqueeze(1), red, best_rows)
        steps += 1

        if steps % sync_every != 0 and steps != max_steps:
            continue
        torch.cuda.synchronize(target_device)
        elapsed = time.perf_counter() - started
        best_value = int(
            (best_k5 if objective == OBJECTIVE_K5 else best_k4).min().item()
        )
        if elapsed >= next_report:
            flips_per_second = steps * batch_size / max(elapsed, 1e-12)
            print(
                f"steps={steps} chains={batch_size} best={best_value} "
                f"aggregate_flips_per_second={flips_per_second:.0f}",
                flush=True,
            )
            next_report += report_every_seconds
        if objective == OBJECTIVE_K5 and best_value == 0:
            stop_reason = "zero_k5"
            break
        if elapsed >= seconds:
            stop_reason = "seconds"
            break

    torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - started
    metric = best_k5 if objective == OBJECTIVE_K5 else best_k4
    best_index = int(torch.argmin(metric).item())
    tracked_k5 = int(best_k5[best_index].item())
    tracked_k4 = int(best_k4[best_index].item())
    selected_red = [int(row) for row in best_rows[best_index].cpu().tolist()]

    verified_k5 = count_monochromatic_cliques(n, selected_red, 5)
    verified_k4 = count_monochromatic_cliques(n, selected_red, 4)
    if verified_k5 != tracked_k5:
        raise RuntimeError(
            "refusing GPU result: tracked K5 cost disagrees with the plain Python recount"
        )
    if objective == OBJECTIVE_K4_AT_ZERO_K5:
        if verified_k5 != 0:
            raise RuntimeError("refusing GPU result: the hard K5-free constraint was violated")
        if verified_k4 != tracked_k4:
            raise RuntimeError(
                "refusing GPU result: tracked K4 cost disagrees with the plain Python recount"
            )

    certificate_path = None
    if verified_k5 == 0 and output_dir is not None:
        certificate_path = str(_write_certificate(output_dir, n, selected_red))
    aggregate_flips = steps * batch_size
    aggregate_rate = aggregate_flips / max(elapsed, 1e-12)
    return SearchResult(
        objective=objective,
        n=n,
        batch_size=batch_size,
        seed=seed,
        steps=steps,
        aggregate_flips=aggregate_flips,
        elapsed_seconds=elapsed,
        aggregate_flips_per_second=aggregate_rate,
        provided_cpu_reference_flips_per_second=CPU_REFERENCE_FLIPS_PER_SECOND,
        multiple_of_provided_cpu_reference=aggregate_rate / CPU_REFERENCE_FLIPS_PER_SECOND,
        verified_monochromatic_k5s=verified_k5,
        verified_monochromatic_k4s=verified_k4,
        stop_reason=stop_reason,
        certificate_path=certificate_path,
    )


def main() -> None:
    """Run the annealer directly inside a CUDA container."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=43)
    parser.add_argument("--objective", choices=OBJECTIVES, default=OBJECTIVE_K5)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-temperature", type=float)
    parser.add_argument("--end-temperature", type=float)
    parser.add_argument("--edge-chunk", type=int, default=128)
    parser.add_argument("--sync-every", type=int, default=10)
    parser.add_argument("--report-every-seconds", type=float, default=30.0)
    args = parser.parse_args()
    result = search(
        n=args.n,
        objective=args.objective,
        batch_size=args.batch_size,
        seconds=args.seconds,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        start_path=args.start,
        output_dir=args.output_dir,
        start_temperature=args.start_temperature,
        end_temperature=args.end_temperature,
        edge_chunk=args.edge_chunk,
        sync_every=args.sync_every,
        report_every_seconds=args.report_every_seconds,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
