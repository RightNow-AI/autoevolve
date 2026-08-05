"""Exact simulated annealing for multicolor Ramsey lower-bound certificates.

Direct mode recolors one edge at a time. Circulant mode colors edges by their
cyclic difference in Z_n and recolors one complete difference class at a time.
Both modes track the exact count of all four forbidden subgraph families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

RED = 0
BLUE = 1
GREEN = 2
YELLOW = 3
COLOR_NAMES = ("red_K3", "blue_K4", "green_C4", "yellow_C4")
MODES = ("direct", "circulant")

Edge = tuple[int, int]
ViolationCounts = tuple[int, int, int, int]
CertificateCallback = Callable[[ViolationCounts, list[int]], None]


@dataclass
class ColoringState:
    """One complete coloring and its exact incremental bookkeeping."""

    n: int
    colors: list[int]
    adjacency: list[list[int]]
    counts: list[int]


@dataclass(frozen=True)
class EdgeMove:
    """One edge recoloring with an exact component-wise cost delta."""

    edge: Edge
    new_color: int
    delta: ViolationCounts

    @property
    def total_delta(self) -> int:
        return sum(self.delta)


@dataclass(frozen=True)
class ClassProposal:
    """One exact circulant difference-class recoloring proposal."""

    difference: int
    new_color: int
    state: ColoringState
    delta: ViolationCounts

    @property
    def total_delta(self) -> int:
        return sum(self.delta)


@dataclass(frozen=True)
class SearchResult:
    """Retained best exact state and search diagnostics."""

    mode: str
    n: int
    seed: int
    violations: ViolationCounts
    colors: list[int]
    steps: int
    restarts: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def edge_index(n: int, left: int, right: int) -> int:
    """Return the lexicographic complete-graph index of one undirected edge."""

    if left > right:
        left, right = right, left
    return left * (2 * n - left - 1) // 2 + right - left - 1


def edge_color(colors: list[int], n: int, left: int, right: int) -> int:
    return colors[edge_index(n, left, right)]


def validate_colors(n: int, colors: list[int]) -> None:
    """Reject any object that is not one color for every pair."""

    expected = n * (n - 1) // 2
    if n < 5:
        raise ValueError("n must be at least five")
    if len(colors) != expected:
        raise ValueError(f"expected {expected} colors, got {len(colors)}")
    for index, color in enumerate(colors):
        if not isinstance(color, int) or isinstance(color, bool) or not 0 <= color <= 3:
            raise ValueError(f"color {index} must be a plain integer in 0..3")


def build_adjacency(n: int, colors: list[int]) -> list[list[int]]:
    """Build four symmetric adjacency bitset arrays from the complete snapshot."""

    validate_colors(n, colors)
    adjacency = [[0] * n for _ in range(4)]
    for left, right in combinations(range(n), 2):
        color = edge_color(colors, n, left, right)
        adjacency[color][left] |= 1 << right
        adjacency[color][right] |= 1 << left
    return adjacency


def full_recount(n: int, colors: list[int]) -> ViolationCounts:
    """Enumerate every forbidden subgraph from scratch with exact integers."""

    validate_colors(n, colors)
    red_triangles = 0
    for vertices in combinations(range(n), 3):
        if all(
            edge_color(colors, n, left, right) == RED
            for left, right in combinations(vertices, 2)
        ):
            red_triangles += 1

    blue_k4 = 0
    green_c4 = 0
    yellow_c4 = 0
    for vertices in combinations(range(n), 4):
        if all(
            edge_color(colors, n, left, right) == BLUE
            for left, right in combinations(vertices, 2)
        ):
            blue_k4 += 1
        first, second, third, fourth = vertices
        cycles = (
            (first, second, third, fourth),
            (first, second, fourth, third),
            (first, third, second, fourth),
        )
        for cycle in cycles:
            cycle_colors = [
                edge_color(colors, n, cycle[index], cycle[(index + 1) % 4])
                for index in range(4)
            ]
            if all(color == GREEN for color in cycle_colors):
                green_c4 += 1
            if all(color == YELLOW for color in cycle_colors):
                yellow_c4 += 1
    return red_triangles, blue_k4, green_c4, yellow_c4


def state_from_colors(n: int, colors: list[int]) -> ColoringState:
    snapshot = list(colors)
    return ColoringState(
        n=n,
        colors=snapshot,
        adjacency=build_adjacency(n, snapshot),
        counts=list(full_recount(n, snapshot)),
    )


def copy_state(state: ColoringState) -> ColoringState:
    return ColoringState(
        n=state.n,
        colors=list(state.colors),
        adjacency=[list(rows) for rows in state.adjacency],
        counts=list(state.counts),
    )


def _edges_in(mask: int, adjacency: list[int]) -> int:
    total = 0
    remaining = mask
    while remaining:
        vertex_bit = remaining & -remaining
        remaining ^= vertex_bit
        vertex = vertex_bit.bit_length() - 1
        total += (adjacency[vertex] & mask).bit_count()
    return total // 2


def _three_paths_between(left: int, right: int, adjacency: list[int]) -> int:
    """Count exact color paths left-a-b-right with distinct internal vertices."""

    blocked = (1 << left) | (1 << right)
    first_steps = adjacency[left] & ~blocked
    last_steps = adjacency[right] & ~blocked
    total = 0
    while first_steps:
        first_bit = first_steps & -first_steps
        first_steps ^= first_bit
        first = first_bit.bit_length() - 1
        total += (adjacency[first] & last_steps & ~blocked).bit_count()
    return total


def _through_edge(state: ColoringState, edge: Edge, color: int) -> int:
    """Count forbidden structures of one color that would contain the edge."""

    left, right = edge
    adjacency = state.adjacency[color]
    common = adjacency[left] & adjacency[right]
    if color == RED:
        return common.bit_count()
    if color == BLUE:
        return _edges_in(common, adjacency)
    return _three_paths_between(left, right, adjacency)


def measure_move(state: ColoringState, edge: Edge, new_color: int) -> EdgeMove:
    """Measure one exact recoloring without changing the state."""

    left, right = edge
    old_color = edge_color(state.colors, state.n, left, right)
    if new_color == old_color or not 0 <= new_color <= 3:
        raise ValueError("new color must differ from the current color and be in 0..3")
    delta = [0, 0, 0, 0]
    delta[old_color] -= _through_edge(state, edge, old_color)
    delta[new_color] += _through_edge(state, edge, new_color)
    return EdgeMove(edge=edge, new_color=new_color, delta=tuple(delta))


def apply_move(state: ColoringState, move: EdgeMove) -> None:
    """Apply one measured move and its exact component delta."""

    left, right = move.edge
    index = edge_index(state.n, left, right)
    old_color = state.colors[index]
    if old_color == move.new_color:
        raise ValueError("cannot apply a recoloring to the same color")
    bit_left = 1 << left
    bit_right = 1 << right
    state.adjacency[old_color][left] ^= bit_right
    state.adjacency[old_color][right] ^= bit_left
    state.adjacency[move.new_color][left] ^= bit_right
    state.adjacency[move.new_color][right] ^= bit_left
    state.colors[index] = move.new_color
    for color, change in enumerate(move.delta):
        state.counts[color] += change
        if state.counts[color] < 0:
            raise RuntimeError("incremental violation count became negative")


def random_colors(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(4) for _ in range(n * (n - 1) // 2)]


def random_edge(n: int, rng: random.Random) -> Edge:
    left = rng.randrange(n)
    right = rng.randrange(n - 1)
    if right >= left:
        right += 1
    return (left, right) if left < right else (right, left)


def random_new_color(old_color: int, rng: random.Random) -> int:
    choice = rng.randrange(3)
    return choice if choice < old_color else choice + 1


def choose_edge_move(
    state: ColoringState,
    rng: random.Random,
    samples: int = 16,
) -> EdgeMove:
    """Choose the best exact delta from a small random tournament."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    moves: list[EdgeMove] = []
    seen: set[tuple[Edge, int]] = set()
    while len(moves) < samples:
        edge = random_edge(state.n, rng)
        old_color = edge_color(state.colors, state.n, *edge)
        new_color = random_new_color(old_color, rng)
        key = edge, new_color
        if key in seen:
            continue
        seen.add(key)
        moves.append(measure_move(state, edge, new_color))
    best_delta = min(move.total_delta for move in moves)
    return rng.choice([move for move in moves if move.total_delta == best_delta])


def choose_plateau_move(
    state: ColoringState,
    rng: random.Random,
    attempts: int = 96,
) -> EdgeMove | None:
    """Find a zero-delta move that crosses the current flat basin exactly."""

    for _ in range(attempts):
        edge = random_edge(state.n, rng)
        old_color = edge_color(state.colors, state.n, *edge)
        move = measure_move(state, edge, random_new_color(old_color, rng))
        if move.total_delta == 0:
            return move
    return None


def _perturb_direct(
    n: int,
    best_colors: list[int],
    rng: random.Random,
    restart_index: int,
) -> ColoringState:
    if restart_index % 4 == 0:
        return state_from_colors(n, random_colors(n, rng))
    state = state_from_colors(n, best_colors)
    changed: set[tuple[Edge, int]] = set()
    while len(changed) < n:
        edge = random_edge(n, rng)
        old_color = edge_color(state.colors, n, *edge)
        new_color = random_new_color(old_color, rng)
        key = edge, new_color
        if key in changed:
            continue
        changed.add(key)
        apply_move(state, measure_move(state, edge, new_color))
    return state


def _report(
    mode: str,
    n: int,
    seed: int,
    elapsed: float,
    state: ColoringState,
    best_counts: ViolationCounts,
    steps: int,
    restarts: int,
) -> None:
    rate = steps / max(elapsed, 1e-9)
    print(
        f"mode={mode} n={n} seed={seed} t={elapsed:.0f}s "
        f"cost={sum(state.counts)} best={sum(best_counts)} "
        f"parts={tuple(state.counts)} restarts={restarts} moves/s={rate:.1f}",
        flush=True,
    )


def _search_direct(
    n: int,
    seed: int,
    seconds: float,
    on_certificate: CertificateCallback | None,
) -> SearchResult:
    rng = random.Random(seed)
    state = state_from_colors(n, random_colors(n, rng))
    best_counts = tuple(state.counts)
    best_colors = list(state.colors)
    started = time.monotonic()
    if sum(best_counts) == 0:
        if on_certificate is not None:
            on_certificate(best_counts, list(best_colors))
        return SearchResult(
            mode="direct",
            n=n,
            seed=seed,
            violations=best_counts,
            colors=best_colors,
            steps=0,
            restarts=0,
            elapsed_seconds=time.monotonic() - started,
        )
    deadline = started + seconds
    last_improvement = started
    next_report = started
    stagnation_window = max(15.0, min(180.0, seconds / 5.0))
    plateau_after = max(500, n * n)
    last_improvement_step = 0
    steps = 0
    restarts = 0

    while (now := time.monotonic()) < deadline:
        if now >= next_report:
            _report(
                "direct",
                n,
                seed,
                now - started,
                state,
                best_counts,
                steps,
                restarts,
            )
            next_report = now + 60.0

        if now - last_improvement >= stagnation_window:
            restarts += 1
            state = _perturb_direct(n, best_colors, rng, restarts)
            last_improvement = now
            last_improvement_step = steps
            restarted = tuple(state.counts)
            if sum(restarted) < sum(best_counts):
                best_counts = restarted
                best_colors = list(state.colors)
                if sum(best_counts) == 0:
                    if on_certificate is not None:
                        on_certificate(best_counts, list(best_colors))
                    break
            continue

        steps += 1
        move = choose_edge_move(state, rng)
        if steps - last_improvement_step >= plateau_after:
            plateau = choose_plateau_move(state, rng)
            if plateau is not None:
                move = plateau

        epoch_fraction = min(1.0, (now - last_improvement) / stagnation_window)
        temperature = max(0.05, 4.0 * math.exp(-4.0 * epoch_fraction))
        accepted = move.total_delta <= 0 or rng.random() < math.exp(
            -move.total_delta / temperature
        )
        if not accepted:
            continue
        apply_move(state, move)
        current = tuple(state.counts)
        if sum(current) < sum(best_counts):
            best_counts = current
            best_colors = list(state.colors)
            last_improvement = now
            last_improvement_step = steps
            if sum(best_counts) == 0:
                if on_certificate is not None:
                    on_certificate(best_counts, list(best_colors))
                break

    elapsed = time.monotonic() - started
    return SearchResult(
        mode="direct",
        n=n,
        seed=seed,
        violations=best_counts,
        colors=best_colors,
        steps=steps,
        restarts=restarts,
        elapsed_seconds=elapsed,
    )


def circulant_edges(n: int, difference: int) -> tuple[Edge, ...]:
    """Return each undirected edge of one cyclic difference class once."""

    if not 1 <= difference <= n // 2:
        raise ValueError("difference is outside the circulant class range")
    edges = {
        tuple(sorted((vertex, (vertex + difference) % n)))
        for vertex in range(n)
    }
    return tuple(sorted(edges))


def colors_from_classes(n: int, class_colors: list[int]) -> list[int]:
    expected = n // 2
    if len(class_colors) != expected:
        raise ValueError(f"expected {expected} circulant class colors")
    colors: list[int] = []
    for left, right in combinations(range(n), 2):
        raw = right - left
        difference = min(raw, n - raw)
        colors.append(class_colors[difference - 1])
    return colors


def random_class_colors(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(4) for _ in range(n // 2)]


def measure_class_proposal(
    state: ColoringState,
    difference: int,
    new_color: int,
) -> ClassProposal:
    """Apply one class recoloring to a copy through exact sequential deltas."""

    proposal = copy_state(state)
    before = tuple(state.counts)
    for edge in circulant_edges(state.n, difference):
        old_color = edge_color(proposal.colors, proposal.n, *edge)
        if old_color == new_color:
            continue
        apply_move(proposal, measure_move(proposal, edge, new_color))
    delta = tuple(after - prior for after, prior in zip(proposal.counts, before, strict=True))
    return ClassProposal(
        difference=difference,
        new_color=new_color,
        state=proposal,
        delta=delta,
    )


def choose_class_proposal(
    state: ColoringState,
    class_colors: list[int],
    rng: random.Random,
    samples: int = 6,
) -> ClassProposal:
    """Choose the best measured difference-class recoloring tournament."""

    proposals: list[ClassProposal] = []
    seen: set[tuple[int, int]] = set()
    target = min(samples, (state.n // 2) * 3)
    while len(proposals) < target:
        difference = rng.randrange(1, state.n // 2 + 1)
        old_color = class_colors[difference - 1]
        new_color = random_new_color(old_color, rng)
        key = difference, new_color
        if key in seen:
            continue
        seen.add(key)
        proposals.append(measure_class_proposal(state, difference, new_color))
    best_delta = min(proposal.total_delta for proposal in proposals)
    return rng.choice(
        [proposal for proposal in proposals if proposal.total_delta == best_delta]
    )


def _perturb_classes(
    n: int,
    best_classes: list[int],
    rng: random.Random,
    restart_index: int,
) -> tuple[list[int], ColoringState]:
    if restart_index % 4 == 0:
        classes = random_class_colors(n, rng)
    else:
        classes = list(best_classes)
        changes = min(3, len(classes))
        for difference in rng.sample(range(len(classes)), changes):
            classes[difference] = random_new_color(classes[difference], rng)
    return classes, state_from_colors(n, colors_from_classes(n, classes))


def _search_circulant(
    n: int,
    seed: int,
    seconds: float,
    on_certificate: CertificateCallback | None,
) -> SearchResult:
    rng = random.Random(seed)
    class_colors = random_class_colors(n, rng)
    state = state_from_colors(n, colors_from_classes(n, class_colors))
    best_counts = tuple(state.counts)
    best_colors = list(state.colors)
    best_classes = list(class_colors)
    started = time.monotonic()
    if sum(best_counts) == 0:
        if on_certificate is not None:
            on_certificate(best_counts, list(best_colors))
        return SearchResult(
            mode="circulant",
            n=n,
            seed=seed,
            violations=best_counts,
            colors=best_colors,
            steps=0,
            restarts=0,
            elapsed_seconds=time.monotonic() - started,
        )
    deadline = started + seconds
    last_improvement = started
    next_report = started
    stagnation_window = max(30.0, min(240.0, seconds / 4.0))
    steps = 0
    restarts = 0

    while (now := time.monotonic()) < deadline:
        if now >= next_report:
            _report(
                "circulant",
                n,
                seed,
                now - started,
                state,
                best_counts,
                steps,
                restarts,
            )
            next_report = now + 60.0

        if now - last_improvement >= stagnation_window:
            restarts += 1
            class_colors, state = _perturb_classes(
                n,
                best_classes,
                rng,
                restarts,
            )
            last_improvement = now
            restarted = tuple(state.counts)
            if sum(restarted) < sum(best_counts):
                best_counts = restarted
                best_colors = list(state.colors)
                best_classes = list(class_colors)
                if sum(best_counts) == 0:
                    if on_certificate is not None:
                        on_certificate(best_counts, list(best_colors))
                    break
            continue

        steps += 1
        proposal = choose_class_proposal(state, class_colors, rng)
        epoch_fraction = min(1.0, (now - last_improvement) / stagnation_window)
        temperature = max(0.1, 8.0 * math.exp(-4.0 * epoch_fraction))
        accepted = proposal.total_delta <= 0 or rng.random() < math.exp(
            -proposal.total_delta / temperature
        )
        if not accepted:
            continue
        state = proposal.state
        class_colors[proposal.difference - 1] = proposal.new_color
        current = tuple(state.counts)
        if sum(current) < sum(best_counts):
            best_counts = current
            best_colors = list(state.colors)
            best_classes = list(class_colors)
            last_improvement = now
            if sum(best_counts) == 0:
                if on_certificate is not None:
                    on_certificate(best_counts, list(best_colors))
                break

    elapsed = time.monotonic() - started
    return SearchResult(
        mode="circulant",
        n=n,
        seed=seed,
        violations=best_counts,
        colors=best_colors,
        steps=steps,
        restarts=restarts,
        elapsed_seconds=elapsed,
    )


def search(
    n: int,
    seed: int,
    seconds: float,
    mode: str = "direct",
    on_certificate: CertificateCallback | None = None,
) -> SearchResult:
    """Run one bounded exact-cost search and return its retained best state."""

    if n < 5:
        raise ValueError("n must be at least five")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if mode == "direct":
        return _search_direct(n, seed, seconds, on_certificate)
    if mode == "circulant":
        return _search_circulant(n, seed, seconds, on_certificate)
    raise ValueError(f"mode must be one of {MODES}")


def write_verified_certificate(
    path: Path,
    n: int,
    colors: list[int],
    expected: ViolationCounts,
) -> tuple[ViolationCounts, Path]:
    """Recount all violations and atomically write only a valid certificate."""

    recounted = full_recount(n, colors)
    if recounted != expected:
        raise ValueError(
            "incremental violation totals disagreed with the full recount: "
            f"expected {expected}, recounted {recounted}"
        )
    if sum(recounted) != 0:
        details = ", ".join(
            f"{name}={value}" for name, value in zip(COLOR_NAMES, recounted, strict=True)
        )
        raise ValueError(f"refusing to write a coloring with violations: {details}")

    payload = {"n": n, "edge_colors": list(colors)}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    if path.suffix:
        target = path
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"n{n}-{digest}.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if target.is_file():
        if target.read_bytes() != encoded:
            raise ValueError(f"certificate target already contains different bytes: {target}")
        return recounted, target
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    return recounted, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--mode", choices=MODES, default="direct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = search(args.n, args.seed, args.seconds, args.mode)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
    if sum(result.violations) == 0:
        output = args.output or Path(
            f"multiramsey-n{args.n}-{args.mode}-seed{args.seed}.json"
        )
        recounted, target = write_verified_certificate(
            output,
            args.n,
            result.colors,
            result.violations,
        )
        print(f"recounted violations: {recounted}", flush=True)
        print(f"certificate written: {target}", flush=True)


if __name__ == "__main__":
    main()
