"""Rich terminal dashboard that watches a run through read-only SQLite queries."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoevolve.cli._data import (
    Snapshot,
    artifact_paths,
    format_number,
    humanize_event,
    load_snapshot,
    terminal_reason,
    why_ended,
)


def watch_run(
    home: Path,
    run_id: str,
    *,
    refresh: float = 1.0,
    render_live: bool = False,
    out_dir: Path,
) -> None:
    """Display live database state until the run closes or the user interrupts."""

    next_render = 0.0
    try:
        initial = load_snapshot(home, run_id)
        with Live(
            build_watch_view(initial, out_dir),
            refresh_per_second=max(1.0, min(20.0, 1.0 / refresh)),
            screen=False,
        ) as live:
            while True:
                snapshot = load_snapshot(home, run_id)
                now = time.monotonic()
                if render_live and now >= next_render:
                    from autoevolve.cli.render import render_all

                    render_all(home, run_id, out_dir, live=True)
                    next_render = now + max(2.0, refresh)
                live.update(build_watch_view(snapshot, out_dir), refresh=True)
                if terminal_reason(snapshot) != "open":
                    break
                time.sleep(refresh)
    except KeyboardInterrupt:
        return


def build_watch_view(snapshot: Snapshot, out_dir: Path) -> Panel:
    """Build one deterministic Rich view from a loaded snapshot."""

    curve = snapshot.curve()
    best = snapshot.best_program()
    budget = snapshot.run.budget
    max_evals = budget.get("max_evals")
    if max_evals is not None:
        budget_text = f"{snapshot.eval_count}/{format_number(max_evals)} evaluations"
    elif budget.get("wall_clock_s") is not None:
        budget_text = (
            f"{format_number(snapshot.elapsed_seconds())}/"
            f"{format_number(budget['wall_clock_s'])} seconds"
        )
    else:
        budget_text = f"{snapshot.eval_count} evaluations"
    idle, plateau_n, reached = snapshot.plateau_state()
    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(justify="right")
    header.add_row(
        Text(f"Run {snapshot.run.id}", style="bold"),
        Text(terminal_reason(snapshot), style="bold cyan"),
    )
    header.add_row(Text(snapshot.run.goal_text), Text(budget_text))
    header.add_row(
        Text(
            f"Best: {best.id if best else 'none'} at "
            f"{format_number(snapshot.score(best.id) if best else None)}"
        ),
        Text(f"Plateau: {idle}/{plateau_n}{' reached' if reached else ''}"),
    )
    header.add_row(
        Text(f"Curve: {sparkline([point.value for point in curve])}"),
        Text(snapshot.metric),
    )

    islands = Table(title="Islands", expand=True)
    islands.add_column("island", justify="right")
    islands.add_column("evals", justify="right")
    islands.add_column("best program")
    islands.add_column("best score", justify="right")
    for summary in snapshot.island_summaries():
        islands.add_row(
            str(summary.island),
            str(summary.evals),
            summary.best_program_id or "none",
            format_number(summary.best_value),
        )

    events = Table(title="Latest events", expand=True)
    events.add_column("seq", justify="right", style="dim")
    events.add_column("time", style="dim")
    events.add_column("event")
    for event in snapshot.events[-8:]:
        events.add_row(str(event.seq), event.created_at, humanize_event(event))
    if not snapshot.events:
        events.add_row("-", "-", "No events recorded")

    paths = artifact_paths(out_dir)
    artifacts = Text()
    artifacts.append("Artifacts\n", style="bold")
    artifacts.append(f"dashboard: {paths['dashboard']}\n")
    artifacts.append(f"gif: {paths['gif']}\n")
    artifacts.append(f"mp4: {paths['mp4']}\n")
    artifacts.append(f"poster: {paths['poster_png']}\n")
    artifacts.append(f"report: {paths['report']}\n")
    artifacts.append(f"latest: {paths['latest_png']}")
    reason = Text(why_ended(snapshot), style="dim")
    return Panel(
        Group(header, islands, events, reason, artifacts),
        title="autoevolve watch",
        border_style="blue",
        padding=(1, 2),
    )


def sparkline(values: list[float]) -> str:
    """Render a compact unicode sparkline for measured best values."""

    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    low = min(values)
    high = max(values)
    if high <= low:
        return blocks[3] * len(values)
    return "".join(
        blocks[min(len(blocks) - 1, int((value - low) / (high - low) * len(blocks)))]
        for value in values
    )
