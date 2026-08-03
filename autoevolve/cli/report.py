"""Markdown run reports reconstructed entirely from the SQLite store."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from autoevolve import __version__
from autoevolve.cli._data import (
    Snapshot,
    artifact_paths,
    format_number,
    load_snapshot,
    terminal_reason,
    why_ended,
)


def report(home: Path, run_id: str, out_path: Path) -> Path:
    """Write the durable Markdown report for one run."""

    snapshot = load_snapshot(home, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(out_path.parent)
    contract_json = json.dumps(snapshot.contract, indent=2, sort_keys=True)
    baseline_id, baseline_value = _measured_baseline(snapshot)
    best = snapshot.best_program()
    best_scores = snapshot.scores.get(best.id, {}) if best else {}
    best_lines = [
        f"- Program: `{best.id if best else 'none'}`",
        (
            f"- Primary metric `{snapshot.metric}`: "
            f"{format_number(snapshot.score(best.id) if best else None)}"
        ),
    ]
    best_lines.extend(
        f"- Score `{metric}`: {format_number(value)}"
        for metric, value in sorted(best_scores.items())
        if metric != snapshot.metric
    )
    milestone_rows = _milestone_rows(snapshot)
    artifact_lines = _artifact_lines(paths)
    content = f"""# autoevolve run report: {snapshot.run.id}

## Outcome

- Status: `{terminal_reason(snapshot)}`
- Domain: `{snapshot.run.domain}`
- Goal: {snapshot.run.goal_text}

{why_ended(snapshot)}

## Locked contract

```json
{contract_json}
```

## Measured baseline

- Program: `{baseline_id}`
- `{snapshot.metric}`: {format_number(baseline_value)}

## Best found

{chr(10).join(best_lines)}

## Fitness milestones

| evaluation | program | best {snapshot.metric} |
|---:|---|---:|
{milestone_rows}

## Artifacts

{artifact_lines}

## Replay

Replay run `{snapshot.run.id}` with recorded seed `{snapshot.run.seed}`. The database contains the
ordered programs, scores, lineage edges, and append-only events needed to reconstruct this result.

## Version

`autoevolve {__version__}`
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _measured_baseline(snapshot: Snapshot) -> tuple[str, float | None]:
    seeds = [program for program in snapshot.programs if program.operator == "seed"]
    if seeds:
        seed = seeds[0]
        measured = snapshot.score(seed.id)
        if measured is not None:
            return seed.id, measured
    value = snapshot.contract.get("baseline")
    baseline = float(value) if isinstance(value, int | float) else None
    return "contract", baseline


def _milestone_rows(snapshot: Snapshot) -> str:
    milestones = snapshot.milestones()
    if not milestones:
        return "| 0 | none | not measured |"
    return "\n".join(
        f"| {point.eval_idx} | `{point.program_id}` | {format_number(point.value)} |"
        for point in milestones
    )


def _artifact_lines(paths: dict[str, Path]) -> str:
    lines = []
    labels = {
        "dashboard": "Dashboard",
        "gif": "Evolution GIF",
        "mp4": "Evolution MP4",
        "poster_png": "Lineage poster PNG",
        "poster_svg": "Lineage poster SVG",
        "report": "Report",
    }
    for key in ("dashboard", "gif", "mp4", "poster_svg", "poster_png", "report"):
        path = paths[key]
        if key == "mp4" and not path.exists() and shutil.which("ffmpeg") is None:
            detail = "not generated because ffmpeg is unavailable"
        elif path.exists() or key == "report":
            detail = str(path.resolve())
        else:
            detail = f"{path.resolve()} (not generated yet)"
        lines.append(f"- {labels[key]}: `{detail}`")
    return "\n".join(lines)
