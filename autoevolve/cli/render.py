"""Deterministic lineage, animation, and poster rendering for autoevolve runs."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps  # noqa: E402
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from autoevolve.cli._data import (  # noqa: E402
    ProgramRow,
    Snapshot,
    artifact_paths,
    event_program_id,
    format_number,
    load_snapshot,
)

ACCENT = "#e97828"
EDGE_COLOR = "#a7adb5"
FAILED_COLOR = "#8a9099"
TEXT_COLOR = "#23272f"
PANEL_COLOR = "#f6f7f9"
MARKERS = {
    "agentic": "^",
    "crossover": "D",
    "diff": "o",
    "rewrite": "s",
    "seed": "o",
}
MAX_FRAMES = 120


def compute_layout(
    programs: list[tuple[Any, ...]],
    edges: list[tuple[Any, ...]],
) -> dict[str, tuple[float, float]]:
    """Compute an incremental layered DAG layout with one vertical lane per island.

    Program tuples use the normative programs-table order. Existing positions depend
    only on earlier rows, so passing a longer suffix can never move an existing node.
    """

    parsed = [_program_tuple(program) for program in programs]
    edge_parents: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if len(edge) >= 2:
            edge_parents[str(edge[0])].append(str(edge[1]))
    layout: dict[str, tuple[float, float]] = {}
    depths: dict[str, int] = {}
    layer_counts: dict[tuple[int, int], int] = defaultdict(int)
    for program_id, parent_id, island in parsed:
        parent_ids: list[str] = []
        if parent_id is not None:
            parent_ids.append(parent_id)
        parent_ids.extend(edge_parents.get(program_id, []))
        known_depths = [depths[parent] for parent in parent_ids if parent in depths]
        depth = max(known_depths, default=-1) + 1
        rank = layer_counts[(island, depth)]
        layer_counts[(island, depth)] += 1
        offset = 0.0 if rank == 0 else 0.42 * math.sin(rank * 2.399963229728653)
        layout[program_id] = (float(island) * 2.0 + offset, float(depth))
        depths[program_id] = depth
    return layout


def render_all(
    home: Path,
    run_id: str,
    out_dir: Path,
    live: bool = False,
) -> dict[str, Path | None]:
    """Render every public artifact for a run from SQLite state only."""

    snapshot = load_snapshot(home, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(out_dir)
    programs = [program.as_tuple() for program in snapshot.programs]
    edges = list(snapshot.edges)
    layout = compute_layout(programs, edges)
    frame_evals = _sample_frame_evals(snapshot)
    frames: list[Image.Image] = []
    try:
        for frame_number, eval_count in enumerate(frame_evals, start=1):
            frames.append(
                _render_frame(
                    snapshot,
                    layout,
                    eval_count,
                    frame_number,
                    len(frame_evals),
                )
            )
        if not frames:
            frames.append(_render_frame(snapshot, layout, 0, 1, 1))
        duration_ms = max(80, min(600, 30_000 // len(frames)))
        frames[0].save(
            paths["gif"],
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        if live:
            frames[-1].save(paths["latest_png"], format="PNG", optimize=False)
        mp4_path = _write_mp4(frames, paths["mp4"], duration_ms)
    finally:
        for frame in frames:
            frame.close()
    _render_poster(snapshot, layout, paths["poster_svg"], paths["poster_png"])
    from autoevolve.cli.dashboard import write_dashboard

    write_dashboard(home, run_id, paths["dashboard"], out_dir=out_dir)
    return {
        "gif": paths["gif"],
        "mp4": mp4_path,
        "poster_svg": paths["poster_svg"],
        "poster_png": paths["poster_png"],
        "dashboard": paths["dashboard"],
        "latest_png": paths["latest_png"] if live else None,
    }


def fitness_color(value: float | None, low: float, high: float) -> str:
    """Map a measured fitness value to the fixed single-hue ramp."""

    if value is None:
        return FAILED_COLOR
    normalized = 0.65 if high <= low else (value - low) / (high - low)
    normalized = max(0.16, min(0.95, normalized))
    return str(to_hex(colormaps["Blues"](normalized), keep_alpha=False))


def elite_path(snapshot: Snapshot, best: ProgramRow | None = None) -> tuple[str, ...]:
    """Return the stable primary-parent path from a root to the selected best node."""

    current = best or snapshot.best_program()
    if current is None:
        return ()
    programs = snapshot.program_by_id
    fallback_parents: dict[str, list[str]] = defaultdict(list)
    for child_id, parent_id, _ in snapshot.edges:
        fallback_parents[child_id].append(parent_id)
    reversed_path: list[str] = []
    seen: set[str] = set()
    while current.id not in seen:
        seen.add(current.id)
        reversed_path.append(current.id)
        parent_id = current.parent_id
        if parent_id is None:
            known = [item for item in fallback_parents.get(current.id, []) if item in programs]
            parent_id = known[0] if known else None
        if parent_id is None or parent_id not in programs:
            break
        current = programs[parent_id]
    return tuple(reversed(reversed_path))


def _program_tuple(program: Sequence[Any]) -> tuple[str, str | None, int]:
    if len(program) < 6:
        raise ValueError("Program rows must contain id, parent_id, and island columns.")
    program_id = str(program[0])
    parent_id = None if program[2] is None else str(program[2])
    return program_id, parent_id, int(program[5])


def _sample_frame_evals(snapshot: Snapshot) -> list[int]:
    total = len(snapshot.submissions)
    if total == 0:
        return [0]
    eval_by_program = {
        program.id: index for index, program in enumerate(snapshot.submissions, start=1)
    }
    improvements: set[int] = set()
    for event in snapshot.events:
        if event.kind != "archive_improved":
            continue
        program_id = event_program_id(event)
        if program_id in eval_by_program:
            improvements.add(eval_by_program[program_id])
            continue
        eval_idx = event.payload.get("eval_idx")
        if isinstance(eval_idx, int) and 0 <= eval_idx <= total:
            improvements.add(eval_idx)
    if not improvements:
        improvements = {point.eval_idx for point in snapshot.milestones() if point.eval_idx > 0}
    required = {0, total, *improvements}
    if len(required) >= MAX_FRAMES:
        ordered = sorted(required)
        return _evenly_select(ordered, MAX_FRAMES)
    frame_budget = min(MAX_FRAMES, max(30, len(required)))
    remaining = frame_budget - len(required)
    interval = max(1, math.ceil(total / max(1, remaining)))
    sampled = required | set(range(interval, total + 1, interval))
    if len(sampled) > frame_budget:
        optional = sorted(sampled - required)
        keep_optional = _evenly_select(optional, frame_budget - len(required))
        sampled = required | set(keep_optional)
    return sorted(sampled)


def _evenly_select(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[-1]]
    indexes = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indexes)]


def _render_frame(
    snapshot: Snapshot,
    layout: dict[str, tuple[float, float]],
    eval_count: int,
    frame_number: int,
    frame_total: int,
) -> Image.Image:
    visible = _visible_programs(snapshot, eval_count)
    best = snapshot.best_program(visible)
    curve = [point for point in snapshot.curve() if point.eval_idx <= eval_count]
    figure, (lineage_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    try:
        figure.patch.set_facecolor("white")
        _draw_lineage(lineage_axis, snapshot, visible, layout, best, poster=False)
        _draw_curve(curve_axis, curve, snapshot)
        figure.suptitle(
            f"Evolution {snapshot.run.id}   frame {frame_number}/{frame_total}   "
            f"evals {eval_count}",
            color=TEXT_COLOR,
            fontsize=11,
            y=0.98,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=100,
            facecolor="white",
            metadata={"Software": "autoevolve"},
        )
        buffer.seek(0)
        with Image.open(buffer) as loaded:
            image = loaded.convert("RGB").copy()
        buffer.close()
        return image
    finally:
        plt.close(figure)


def _visible_programs(snapshot: Snapshot, eval_count: int) -> tuple[ProgramRow, ...]:
    visible_ids = {program.id for program in snapshot.submissions[:eval_count]}
    visible_ids.update(program.id for program in snapshot.programs if program.operator == "seed")
    return tuple(program for program in snapshot.programs if program.id in visible_ids)


def _draw_lineage(
    axis: Any,
    snapshot: Snapshot,
    visible: Sequence[ProgramRow],
    layout: dict[str, tuple[float, float]],
    best: ProgramRow | None,
    *,
    poster: bool,
) -> None:
    visible_ids = {program.id for program in visible}
    path = elite_path(snapshot, best)
    path_edges = set(zip(path[1:], path[:-1], strict=False))
    values = [
        _objective_score(snapshot, snapshot.score(program.id))
        for program in visible
        if snapshot.is_scored(program.id)
    ]
    measured = [value for value in values if value is not None]
    low = min(measured, default=0.0)
    high = max(measured, default=1.0)
    for child_id, parent_id, kind in snapshot.edges:
        if child_id not in visible_ids or parent_id not in visible_ids:
            continue
        child_x, child_y = layout[child_id]
        parent_x, parent_y = layout[parent_id]
        highlighted = (child_id, parent_id) in path_edges
        axis.plot(
            [parent_x, child_x],
            [parent_y, child_y],
            color=ACCENT if highlighted else EDGE_COLOR,
            linewidth=2.3 if highlighted else 0.75,
            linestyle="--" if kind == "migration" else "-",
            alpha=1.0 if highlighted else 0.72,
            zorder=1 if highlighted else 0,
        )
    for program in visible:
        x, y = layout[program.id]
        scored = snapshot.is_scored(program.id)
        marker = MARKERS.get(program.operator, "o")
        axis.scatter(
            [x],
            [y],
            s=42 if poster else 30,
            marker=marker,
            facecolors=(
                fitness_color(_objective_score(snapshot, snapshot.score(program.id)), low, high)
                if scored
                else "none"
            ),
            edgecolors=(ACCENT if program.id in path else FAILED_COLOR),
            linewidths=1.5 if program.id in path else 0.9,
            zorder=3,
        )
    island_ids = sorted({program.island for program in visible})
    for island in island_ids:
        axis.axvline(island * 2.0, color="#e5e7eb", linewidth=0.6, zorder=-1)
    axis.set_title("Lineage", loc="left", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    axis.set_xlabel("island lanes", color=TEXT_COLOR, fontsize=9)
    axis.set_ylabel("lineage depth", color=TEXT_COLOR, fontsize=9)
    axis.set_xticks([island * 2.0 for island in island_ids], [str(island) for island in island_ids])
    axis.tick_params(colors="#69707a", labelsize=8)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#d3d7dc")
    legend = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#5a91c7",
            markeredgecolor="#5a91c7",
            markersize=6,
            label=name,
        )
        for name, marker in (
            ("diff", "o"),
            ("rewrite", "s"),
            ("agentic", "^"),
            ("crossover", "D"),
        )
    ]
    axis.legend(handles=legend, loc="upper left", frameon=False, fontsize=7, ncol=2)
    extent = [layout[program.id] for program in visible] if poster else list(layout.values())
    if extent:
        all_x = [point[0] for point in extent]
        all_y = [point[1] for point in extent]
        axis.set_xlim(min(all_x) - 0.7, max(all_x) + 0.7)
        axis.set_ylim(min(all_y) - 0.7, max(all_y) + 0.7)


def _draw_curve(axis: Any, points: Sequence[Any], snapshot: Snapshot) -> None:
    x_values = [point.eval_idx for point in points]
    y_values = [point.value for point in points]
    if points:
        axis.plot(x_values, y_values, color="#377eb8", linewidth=2.0)
        axis.scatter(x_values[-1:], y_values[-1:], color="#377eb8", s=26, zorder=3)
        axis.annotate(
            format_number(y_values[-1]),
            (x_values[-1], y_values[-1]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_COLOR,
        )
    else:
        axis.text(0.5, 0.5, "No measured scores yet", ha="center", va="center")
    axis.set_title("Best score", loc="left", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    axis.set_xlabel("evaluations", color=TEXT_COLOR, fontsize=9)
    axis.set_ylabel(snapshot.metric, color=TEXT_COLOR, fontsize=9)
    axis.tick_params(colors="#69707a", labelsize=8)
    axis.grid(color="#e3e6ea", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#d3d7dc")
    full_curve = snapshot.curve()
    if full_curve:
        full_x = [point.eval_idx for point in full_curve]
        full_y = [point.value for point in full_curve]
        y_low = min(full_y)
        y_high = max(full_y)
        margin = max(0.05, (y_high - y_low) * 0.08)
        axis.set_xlim(0, max(1, max(full_x)))
        axis.set_ylim(y_low - margin, y_high + margin)


def _write_mp4(frames: Sequence[Image.Image], path: Path, duration_ms: int) -> Path | None:
    executable = shutil.which("ffmpeg")
    if executable is None or not frames:
        return None
    width, height = frames[0].size
    fps = 1000.0 / duration_ms
    command = [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    write_failed = False
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        write_failed = True
    finally:
        process.stdin.close()
    return_code = process.wait()
    if write_failed or return_code != 0:
        path.unlink(missing_ok=True)
        return None
    return path


def _render_poster(
    snapshot: Snapshot,
    layout: dict[str, tuple[float, float]],
    svg_path: Path,
    png_path: Path,
) -> None:
    best = snapshot.best_program()
    selected, capped = _poster_programs(snapshot, best)
    selected_ids = {program.id for program in selected}
    selected_edges = [
        edge
        for edge in snapshot.edges
        if edge[0] in selected_ids and edge[1] in selected_ids
    ]
    poster_snapshot = Snapshot(
        run=snapshot.run,
        programs=tuple(selected),
        edges=tuple(selected_edges),
        scores=snapshot.scores,
        stages=snapshot.stages,
        events=snapshot.events,
        islands=snapshot.islands,
        operators=snapshot.operators,
        discoveries=snapshot.discoveries,
        gate_failed_ids=snapshot.gate_failed_ids,
    )
    figure, axis = plt.subplots(figsize=(13, 8))
    try:
        _draw_lineage(axis, poster_snapshot, selected, layout, best, poster=True)
        path = elite_path(snapshot, best)
        for parent_id, child_id in zip(path, path[1:], strict=False):
            if parent_id not in selected_ids or child_id not in selected_ids:
                continue
            parent_score = snapshot.score(parent_id)
            child_score = snapshot.score(child_id)
            if parent_score is None or child_score is None:
                continue
            parent_x, parent_y = layout[parent_id]
            child_x, child_y = layout[child_id]
            axis.annotate(
                f"{child_score - parent_score:+.4g}",
                ((parent_x + child_x) / 2, (parent_y + child_y) / 2),
                xytext=(4, -4),
                textcoords="offset points",
                fontsize=7,
                color=ACCENT,
                fontweight="bold",
            )
        best_text = "none"
        if best is not None:
            best_text = f"{best.id} at {format_number(snapshot.score(best.id))} {snapshot.metric}"
        title = f"Winning genealogy for {snapshot.run.id}\nBest: {best_text}"
        if capped:
            title += "\nView capped at 600 nodes: elite spine plus 2-hop neighborhood"
        figure.suptitle(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.98)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        matplotlib.rcParams["svg.hashsalt"] = "autoevolve"
        figure.savefig(
            svg_path,
            format="svg",
            facecolor="white",
            metadata={"Creator": "autoevolve", "Date": None},
        )
        figure.savefig(
            png_path,
            format="png",
            dpi=150,
            facecolor="white",
            metadata={"Software": "autoevolve"},
        )
    finally:
        plt.close(figure)


def _poster_programs(
    snapshot: Snapshot,
    best: ProgramRow | None,
) -> tuple[tuple[ProgramRow, ...], bool]:
    if len(snapshot.programs) <= 600:
        return snapshot.programs, False
    path = elite_path(snapshot, best)
    selected = set(path)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for child_id, parent_id, _ in snapshot.edges:
        adjacency[child_id].add(parent_id)
        adjacency[parent_id].add(child_id)
    queue: deque[tuple[str, int]] = deque((program_id, 0) for program_id in path)
    seen = set(path)
    while queue:
        program_id, distance = queue.popleft()
        if distance >= 2:
            continue
        for neighbor in sorted(adjacency.get(program_id, set())):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.add(neighbor)
            queue.append((neighbor, distance + 1))
    if len(selected) > 600:
        if len(path) > 600:
            indexes = _evenly_select(list(range(len(path))), 600)
            path_ids = [path[index] for index in indexes]
        else:
            path_ids = list(path)
        remaining = 600 - len(path_ids)
        neighbors = [
            program.id
            for program in snapshot.programs
            if program.id in selected and program.id not in path_ids
        ]
        selected = set(path_ids + neighbors[: max(0, remaining)])
    ordered = tuple(program for program in snapshot.programs if program.id in selected)
    return ordered, True


def _objective_score(snapshot: Snapshot, value: float | None) -> float | None:
    if value is None:
        return None
    return value if snapshot.maximize else -value
