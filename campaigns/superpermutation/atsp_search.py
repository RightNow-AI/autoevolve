"""Exact asymmetric path search for order-six superpermutations.

The search entry point is disabled unless AUTOEVOLVE_MODAL_ATSP=1. The safe local operation is
independent verification:

python campaigns/superpermutation/atsp_search.py verify superpermutation.txt --reported-length 871
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CostMatrix = NDArray[np.uint8]
PathArray = NDArray[np.int32]

ORDER = 6
DEFAULT_PROGRESS_SECONDS = 60.0
OR_OPT_LENGTHS = (1, 2, 3, 5, 6, 12)


@dataclass(frozen=True, slots=True)
class AtspProblem:
    """The exact permutation graph and its asymmetric append costs."""

    order: int
    permutations: tuple[str, ...]
    costs: CostMatrix
    rotation_cycles: tuple[tuple[int, ...], ...]

    @property
    def node_count(self) -> int:
        return len(self.permutations)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """A total certificate check that does not trust the search path or cost matrix."""

    valid: bool
    order: int
    expected_permutations: int
    found_permutations: int
    actual_length: int
    reported_length: int | None
    length_matches: bool
    alphabet_matches: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result whose length has passed independent exact verification."""

    seed: int
    restarts: int
    elapsed_seconds: float
    path: tuple[int, ...]
    superpermutation: str
    verification: VerificationReport

    def summary_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "restarts": self.restarts,
            "elapsed_seconds": self.elapsed_seconds,
            "length": self.verification.actual_length,
            "verified": self.verification.valid,
        }


@dataclass(slots=True)
class _SearchState:
    path: PathArray
    score: int
    best_path: PathArray
    best_score: int


class VerificationError(RuntimeError):
    """Raised when a candidate cannot be reported honestly."""

    def __init__(self, message: str, report: VerificationReport) -> None:
        super().__init__(message)
        self.report = report


def _symbols(order: int) -> str:
    if order < 2 or order > 9:
        raise ValueError("order must be between 2 and 9")
    return "".join(str(value) for value in range(order))


def _longest_suffix_prefix(a: str, b: str) -> int:
    for overlap in range(min(len(a), len(b)), 0, -1):
        if a[-overlap:] == b[:overlap]:
            return overlap
    return 0


def _append_cost(a: str, b: str) -> int:
    return len(b) - _longest_suffix_prefix(a, b)


def _build_rotation_cycles(
    permutations: tuple[str, ...], index_by_permutation: dict[str, int]
) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    order = len(permutations[0])

    for first_index, first_permutation in enumerate(permutations):
        if first_index in seen:
            continue
        cycle: list[int] = []
        rotated = first_permutation
        for _ in range(order):
            node = index_by_permutation[rotated]
            if node in seen:
                raise RuntimeError("rotation cycles overlap before they close")
            seen.add(node)
            cycle.append(node)
            rotated = rotated[1:] + rotated[:1]
        if rotated != first_permutation:
            raise RuntimeError("rotation cycle did not close")
        cycles.append(tuple(cycle))

    if len(seen) != len(permutations):
        raise RuntimeError("rotation cycles do not cover the permutation graph")
    return tuple(cycles)


def build_problem(order: int = ORDER) -> AtspProblem:
    """Build every node and append cost directly from the overlap definition."""

    symbols = _symbols(order)
    permutations = tuple("".join(items) for items in itertools.permutations(symbols))
    costs = np.empty((len(permutations), len(permutations)), dtype=np.uint8)

    for row, source in enumerate(permutations):
        for column, target in enumerate(permutations):
            costs[row, column] = _append_cost(source, target)

    index_by_permutation = {value: index for index, value in enumerate(permutations)}
    rotation_cycles = _build_rotation_cycles(permutations, index_by_permutation)
    return AtspProblem(
        order=order,
        permutations=permutations,
        costs=costs,
        rotation_cycles=rotation_cycles,
    )


def verify_superpermutation(
    candidate: str, order: int = ORDER, reported_length: int | None = None
) -> VerificationReport:
    """Verify every required substring without using a path or an ATSP score."""

    symbols = _symbols(order)
    expected = {"".join(items) for items in itertools.permutations(symbols)}
    windows = {
        candidate[start : start + order]
        for start in range(max(0, len(candidate) - order + 1))
    }
    found = expected.intersection(windows)
    missing = tuple(sorted(expected.difference(found)))
    alphabet_matches = set(candidate).issubset(set(symbols))
    length_matches = reported_length is None or reported_length == len(candidate)
    valid = not missing and alphabet_matches and length_matches
    return VerificationReport(
        valid=valid,
        order=order,
        expected_permutations=len(expected),
        found_permutations=len(found),
        actual_length=len(candidate),
        reported_length=reported_length,
        length_matches=length_matches,
        alphabet_matches=alphabet_matches,
        missing=missing,
    )


def _require_verified(report: VerificationReport, context: str) -> None:
    if report.valid:
        return
    reasons: list[str] = []
    if report.missing:
        reasons.append(
            f"missing {len(report.missing)} of {report.expected_permutations} permutations"
        )
    if not report.alphabet_matches:
        reasons.append("candidate contains symbols outside the required alphabet")
    if not report.length_matches:
        reasons.append("reported length differs from the materialized string length")
    detail = "; ".join(reasons) if reasons else "unknown verification failure"
    raise VerificationError(
        f"VERIFICATION FAILURE in {context}: {detail}. Refusing to report a length.", report
    )


def materialize_path(problem: AtspProblem, path: Sequence[int]) -> str:
    """Materialize a path using direct suffix-prefix comparisons."""

    if len(path) != problem.node_count:
        raise ValueError("path does not contain the required number of nodes")
    nodes = [int(node) for node in path]
    if len(set(nodes)) != problem.node_count:
        raise ValueError("path repeats at least one node")
    if min(nodes) < 0 or max(nodes) >= problem.node_count:
        raise ValueError("path contains a node outside the permutation graph")

    pieces = [problem.permutations[nodes[0]]]
    previous = problem.permutations[nodes[0]]
    for node in nodes[1:]:
        current = problem.permutations[node]
        overlap = _longest_suffix_prefix(previous, current)
        pieces.append(current[overlap:])
        previous = current
    return "".join(pieces)


def _edge_score(problem: AtspProblem, path: PathArray) -> int:
    if len(path) < 2:
        return 0
    return int(np.sum(problem.costs[path[:-1], path[1:]], dtype=np.int64))


def certify_path(
    problem: AtspProblem, path: PathArray, edge_score: int
) -> tuple[str, VerificationReport]:
    """Materialize and independently certify a path before exposing its length."""

    candidate = materialize_path(problem, path)
    reported_length = problem.order + edge_score
    report = verify_superpermutation(candidate, problem.order, reported_length)
    _require_verified(report, "ATSP path certification")
    return candidate, report


class _ProgressReporter:
    def __init__(
        self,
        problem: AtspProblem,
        seed: int,
        started_at: float,
        interval_seconds: float,
    ) -> None:
        self.problem = problem
        self.seed = seed
        self.started_at = started_at
        self.interval_seconds = interval_seconds
        self.last_report_at = started_at - interval_seconds

    def report(self, state: _SearchState, restart: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report_at < self.interval_seconds:
            return
        try:
            _, current_report = certify_path(self.problem, state.path, state.score)
            if np.array_equal(state.path, state.best_path):
                best_report = current_report
            else:
                _, best_report = certify_path(
                    self.problem, state.best_path, state.best_score
                )
        except VerificationError as error:
            print(str(error), file=sys.stderr, flush=True)
            raise

        payload = {
            "event": "atsp_progress",
            "seed": self.seed,
            "restart": restart,
            "elapsed_seconds": round(now - self.started_at, 3),
            "current_length": current_report.actual_length,
            "best_length": best_report.actual_length,
            "verified": True,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        self.last_report_at = now


def _rotated_cycle(cycle: tuple[int, ...], offset: int) -> tuple[int, ...]:
    return cycle[offset:] + cycle[:offset]


def _cycle_greedy_path(problem: AtspProblem, rng: np.random.Generator) -> PathArray:
    """Join complete weight-one rotation paths with randomized restricted choices."""

    remaining = list(range(len(problem.rotation_cycles)))
    first_position = int(rng.integers(0, len(remaining)))
    first_cycle = problem.rotation_cycles[remaining.pop(first_position)]
    first_offset = int(rng.integers(0, problem.order))
    path = list(_rotated_cycle(first_cycle, first_offset))

    while remaining:
        choices: list[tuple[int, float, int, int]] = []
        last_node = path[-1]
        for remaining_position, cycle_index in enumerate(remaining):
            cycle = problem.rotation_cycles[cycle_index]
            for offset in range(problem.order):
                start_node = cycle[offset]
                choices.append(
                    (
                        int(problem.costs[last_node, start_node]),
                        float(rng.random()),
                        remaining_position,
                        offset,
                    )
                )
        choices.sort(key=lambda value: (value[0], value[1]))
        best_entry = choices[0][0]
        competitive = [choice for choice in choices if choice[0] <= best_entry + 1]
        pool = competitive[: min(12, len(competitive))]
        selected = pool[int(rng.integers(0, len(pool)))]
        _, _, remaining_position, offset = selected
        cycle_index = remaining.pop(remaining_position)
        path.extend(_rotated_cycle(problem.rotation_cycles[cycle_index], offset))

    return np.asarray(path, dtype=np.int32)


def _nearest_neighbor_path(problem: AtspProblem, rng: np.random.Generator) -> PathArray:
    """Build a node-level greedy path with randomized ties and near-ties."""

    path = np.empty(problem.node_count, dtype=np.int32)
    unvisited = np.ones(problem.node_count, dtype=np.bool_)
    current = int(rng.integers(0, problem.node_count))
    path[0] = current
    unvisited[current] = False

    for position in range(1, problem.node_count):
        candidates = np.flatnonzero(unvisited)
        row_costs = problem.costs[current, candidates]
        best_cost = int(np.min(row_costs))
        competitive = candidates[row_costs <= best_cost + 1]
        current = int(competitive[int(rng.integers(0, len(competitive)))])
        path[position] = current
        unvisited[current] = False
    return path


def _or_opt_delta(
    problem: AtspProblem, path: PathArray, start: int, length: int, destination: int
) -> int:
    """Return the directed edge delta for relocating one oriented segment."""

    stop = start + length
    reduced_length = len(path) - length
    if destination == start:
        return 0

    costs = problem.costs
    first = int(path[start])
    last = int(path[stop - 1])
    delta = 0

    if start > 0:
        delta -= int(costs[path[start - 1], first])
    if stop < len(path):
        delta -= int(costs[last, path[stop]])
    if start > 0 and stop < len(path):
        delta += int(costs[path[start - 1], path[stop]])

    def reduced_node(index: int) -> int:
        original_index = index if index < start else index + length
        return int(path[original_index])

    if destination > 0 and destination < reduced_length:
        delta -= int(costs[reduced_node(destination - 1), reduced_node(destination)])
    if destination > 0:
        delta += int(costs[reduced_node(destination - 1), first])
    if destination < reduced_length:
        delta += int(costs[last, reduced_node(destination)])
    return delta


def _apply_or_opt(
    path: PathArray, start: int, length: int, destination: int
) -> PathArray:
    stop = start + length
    segment = path[start:stop]
    reduced = np.concatenate((path[:start], path[stop:]))
    return np.concatenate(
        (reduced[:destination], segment, reduced[destination:])
    ).astype(np.int32, copy=False)


def _reversal_prefix(problem: AtspProblem, path: PathArray) -> NDArray[np.int64]:
    forward = problem.costs[path[:-1], path[1:]].astype(np.int64)
    reverse = problem.costs[path[1:], path[:-1]].astype(np.int64)
    prefix = np.zeros(len(path), dtype=np.int64)
    prefix[1:] = np.cumsum(reverse - forward, dtype=np.int64)
    return prefix


def _reversal_delta(
    problem: AtspProblem,
    path: PathArray,
    prefix: NDArray[np.int64],
    start: int,
    stop: int,
) -> int:
    """Return the full asymmetric delta for reversing path[start:stop + 1]."""

    costs = problem.costs
    delta = int(prefix[stop] - prefix[start])
    if start > 0:
        delta -= int(costs[path[start - 1], path[start]])
        delta += int(costs[path[start - 1], path[stop]])
    if stop + 1 < len(path):
        delta -= int(costs[path[stop], path[stop + 1]])
        delta += int(costs[path[start], path[stop + 1]])
    return delta


def _apply_reversal(path: PathArray, start: int, stop: int) -> PathArray:
    candidate = path.copy()
    candidate[start : stop + 1] = path[start : stop + 1][::-1]
    return candidate


def _three_edge_delta(
    problem: AtspProblem, path: PathArray, first: int, second: int, third: int
) -> int:
    """Return the delta for A+B+C+D to A+C+B+D without symmetric assumptions."""

    costs = problem.costs
    delta = -int(costs[path[second - 1], path[second]])
    delta += int(costs[path[third - 1], path[first]])
    if first > 0:
        delta -= int(costs[path[first - 1], path[first]])
        delta += int(costs[path[first - 1], path[second]])
    if third < len(path):
        delta -= int(costs[path[third - 1], path[third]])
        delta += int(costs[path[second - 1], path[third]])
    return delta


def _apply_three_edge(
    path: PathArray, first: int, second: int, third: int
) -> PathArray:
    return np.concatenate(
        (path[:first], path[second:third], path[first:second], path[third:])
    ).astype(np.int32, copy=False)


def _record_best(state: _SearchState) -> None:
    if state.score < state.best_score:
        state.best_score = state.score
        state.best_path = state.path.copy()


def _anneal(
    problem: AtspProblem,
    state: _SearchState,
    rng: np.random.Generator,
    deadline: float,
    reporter: _ProgressReporter,
    restart: int,
) -> None:
    steps = problem.node_count * 160
    initial_temperature = 2.5
    final_temperature = 0.08
    prefix = _reversal_prefix(problem, state.path)

    for step in range(steps):
        if step % 256 == 0:
            reporter.report(state, restart)
            if time.monotonic() >= deadline:
                return

        progress = step / max(1, steps - 1)
        temperature = initial_temperature * math.pow(
            final_temperature / initial_temperature, progress
        )
        move_kind = float(rng.random())
        candidate: PathArray | None = None
        delta = 0

        if move_kind < 0.58:
            length = OR_OPT_LENGTHS[int(rng.integers(0, len(OR_OPT_LENGTHS)))]
            if length >= len(state.path):
                continue
            start = int(rng.integers(0, len(state.path) - length + 1))
            destination = int(rng.integers(0, len(state.path) - length + 1))
            if destination == start:
                continue
            delta = _or_opt_delta(problem, state.path, start, length, destination)
            candidate = _apply_or_opt(state.path, start, length, destination)
        elif move_kind < 0.82:
            spans = (2, 3, 4, 5, 6, 12, 24, 48, 96)
            span = spans[int(rng.integers(0, len(spans)))]
            span = min(span, len(state.path))
            start = int(rng.integers(0, len(state.path) - span + 1))
            stop = start + span - 1
            delta = _reversal_delta(problem, state.path, prefix, start, stop)
            candidate = _apply_reversal(state.path, start, stop)
        else:
            cuts = np.sort(
                rng.choice(np.arange(len(state.path) + 1), size=3, replace=False)
            )
            first, second, third = (int(value) for value in cuts)
            if not (first < second < third):
                continue
            delta = _three_edge_delta(problem, state.path, first, second, third)
            candidate = _apply_three_edge(state.path, first, second, third)

        accept = delta <= 0 or float(rng.random()) < math.exp(-delta / temperature)
        if not accept or candidate is None:
            continue
        state.path = candidate
        state.score += delta
        prefix = _reversal_prefix(problem, state.path)
        _record_best(state)


def _or_opt_descent(
    problem: AtspProblem,
    state: _SearchState,
    deadline: float,
    reporter: _ProgressReporter,
    restart: int,
) -> bool:
    best_delta = 0
    best_move: tuple[int, int, int] | None = None
    for length in (1, 2, 3, 6):
        if length >= len(state.path):
            continue
        reduced_length = len(state.path) - length
        for start in range(len(state.path) - length + 1):
            if start % 24 == 0:
                reporter.report(state, restart)
                if time.monotonic() >= deadline:
                    return False
            for destination in range(reduced_length + 1):
                if destination == start:
                    continue
                delta = _or_opt_delta(
                    problem, state.path, start, length, destination
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (start, length, destination)

    if best_move is None:
        return False
    state.path = _apply_or_opt(state.path, *best_move)
    state.score += best_delta
    _record_best(state)
    return True


def _reversal_descent(
    problem: AtspProblem,
    state: _SearchState,
    deadline: float,
    reporter: _ProgressReporter,
    restart: int,
) -> bool:
    prefix = _reversal_prefix(problem, state.path)
    best_delta = 0
    best_move: tuple[int, int] | None = None
    for start in range(len(state.path) - 1):
        if start % 24 == 0:
            reporter.report(state, restart)
            if time.monotonic() >= deadline:
                return False
        for stop in range(start + 1, len(state.path)):
            delta = _reversal_delta(problem, state.path, prefix, start, stop)
            if delta < best_delta:
                best_delta = delta
                best_move = (start, stop)

    if best_move is None:
        return False
    state.path = _apply_reversal(state.path, *best_move)
    state.score += best_delta
    _record_best(state)
    return True


def _directed_descent(
    problem: AtspProblem,
    state: _SearchState,
    deadline: float,
    reporter: _ProgressReporter,
    restart: int,
) -> None:
    for _ in range(16):
        if time.monotonic() >= deadline:
            return
        if _or_opt_descent(problem, state, deadline, reporter, restart):
            continue
        if _reversal_descent(problem, state, deadline, reporter, restart):
            continue
        return


def search(
    deadline_seconds: float,
    seed: int,
    progress_interval_seconds: float = DEFAULT_PROGRESS_SECONDS,
) -> SearchResult:
    """Search until the wall clock deadline and return only an exact certificate."""

    if os.environ.get("AUTOEVOLVE_MODAL_ATSP") != "1":
        raise RuntimeError("ATSP optimization is disabled outside its Modal CPU image")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    started_at = time.monotonic()
    deadline = started_at + deadline_seconds
    problem = build_problem()
    rng = np.random.default_rng(seed)
    reporter = _ProgressReporter(
        problem=problem,
        seed=seed,
        started_at=started_at,
        interval_seconds=progress_interval_seconds,
    )

    initial_path = _cycle_greedy_path(problem, rng)
    initial_score = _edge_score(problem, initial_path)
    state = _SearchState(
        path=initial_path,
        score=initial_score,
        best_path=initial_path.copy(),
        best_score=initial_score,
    )
    restart = 0
    reporter.report(state, restart, force=True)

    while time.monotonic() < deadline:
        if restart > 0:
            if restart % 4 == 0:
                state.path = _nearest_neighbor_path(problem, rng)
            else:
                state.path = _cycle_greedy_path(problem, rng)
            state.score = _edge_score(problem, state.path)
            _record_best(state)

        _anneal(problem, state, rng, deadline, reporter, restart)
        _directed_descent(problem, state, deadline, reporter, restart)
        if _edge_score(problem, state.path) != state.score:
            raise RuntimeError("incremental ATSP score diverged from the exact edge sum")
        reporter.report(state, restart, force=True)
        restart += 1

    try:
        candidate, report = certify_path(problem, state.best_path, state.best_score)
    except VerificationError as error:
        print(str(error), file=sys.stderr, flush=True)
        raise
    reporter.report(state, restart, force=True)
    return SearchResult(
        seed=seed,
        restarts=restart,
        elapsed_seconds=time.monotonic() - started_at,
        path=tuple(int(node) for node in state.best_path),
        superpermutation=candidate,
        verification=report,
    )


def write_verified_result(result: SearchResult, output_dir: Path) -> None:
    """Write a certificate only after repeating its exact verification."""

    report = verify_superpermutation(
        result.superpermutation,
        order=result.verification.order,
        reported_length=result.verification.actual_length,
    )
    _require_verified(report, "result write")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "superpermutation.txt").write_text(
        result.superpermutation + "\n", encoding="utf-8"
    )
    (output_dir / "verification.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "path.json").write_text(
        json.dumps({"path": result.path}, indent=2) + "\n", encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="independently verify a materialized candidate"
    )
    verify_parser.add_argument("string_file", type=Path)
    verify_parser.add_argument("--reported-length", type=int)

    search_parser = subparsers.add_parser(
        "search", help="run the optimizer inside a marked Modal container"
    )
    search_parser.add_argument("--deadline-seconds", type=float, required=True)
    search_parser.add_argument("--seed", type=int, required=True)
    search_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "verify":
        candidate = args.string_file.read_text(encoding="utf-8").rstrip("\r\n")
        report = verify_superpermutation(
            candidate, order=ORDER, reported_length=args.reported_length
        )
        try:
            _require_verified(report, "standalone verification")
        except VerificationError as error:
            print(str(error), file=sys.stderr)
            return 2
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    if os.environ.get("AUTOEVOLVE_MODAL_ATSP") != "1":
        print(
            "Search is remote-only. Launch it through modal_atsp.py so founder laptop "
            "CPU is unused.",
            file=sys.stderr,
        )
        return 2
    result = search(deadline_seconds=args.deadline_seconds, seed=args.seed)
    write_verified_result(result, args.output_dir)
    print(json.dumps(result.summary_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
