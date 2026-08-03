"""Self-contained HTML dashboard generation for an autoevolve run."""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from typing import Any

from autoevolve.cli._data import (
    Snapshot,
    artifact_paths,
    event_program_id,
    format_number,
    load_snapshot,
    terminal_reason,
    why_ended,
)
from autoevolve.cli.render import ACCENT, EDGE_COLOR, FAILED_COLOR, compute_layout, elite_path


def write_dashboard(
    home: Path,
    run_id: str,
    out_path: Path,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Write one portable dashboard with inline SVG, CSS, and JavaScript."""

    snapshot = load_snapshot(home, run_id)
    target_dir = out_dir or out_path.parent
    paths = artifact_paths(target_dir)
    layout = compute_layout(
        [program.as_tuple() for program in snapshot.programs],
        list(snapshot.edges),
    )
    lineage_svg = _lineage_svg(snapshot, layout)
    curve_svg = _curve_svg(snapshot)
    document = _document(snapshot, lineage_svg, curve_svg, paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    return out_path


def _document(
    snapshot: Snapshot,
    lineage_svg: str,
    curve_svg: str,
    paths: dict[str, Path],
) -> str:
    contract = snapshot.contract
    direction = "maximize" if snapshot.maximize else "minimize"
    budget = snapshot.run.budget
    budget_parts = [
        f"{key}: {format_number(value)}"
        for key, value in budget.items()
        if value is not None
    ]
    contract_items = [
        ("metric", snapshot.metric),
        ("direction", direction),
        ("baseline", format_number(contract.get("baseline"))),
        ("target", format_number(contract.get("target"))),
        ("gate", snapshot.gate),
        ("budget", ", ".join(budget_parts) or "not recorded"),
        ("plateau", format_number(contract.get("plateau_n", 150))),
    ]
    contract_html = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in contract_items
    )
    operator_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td><td>{pulls}</td><td>{improvements}</td>"
        f"<td>{html.escape(format_number(mean_gain))}</td></tr>"
        for name, pulls, improvements, mean_gain in snapshot.operators
    ) or '<tr><td colspan="4" class="muted">No operator updates recorded.</td></tr>'
    island_rows = "".join(
        "<tr>"
        f"<td>{summary.island}</td><td>{summary.evals}</td>"
        f"<td>{html.escape(summary.best_program_id or 'none')}</td>"
        f"<td>{html.escape(format_number(summary.best_value))}</td></tr>"
        for summary in snapshot.island_summaries()
    ) or '<tr><td colspan="4" class="muted">No islands recorded.</td></tr>'
    discovery_items = "".join(
        "<li>"
        f"<span>{html.escape(text)}</span>"
        f"<small>{html.escape(discovery_id)} from "
        f"{html.escape(source_run or 'unknown run')}</small>"
        "</li>"
        for discovery_id, text, source_run, _, _ in snapshot.discoveries
    ) or '<li class="muted">No discoveries recorded for this domain.</li>'
    links = []
    for key in ("gif", "mp4", "poster_svg", "poster_png", "report"):
        path = paths[key]
        label = path.name
        links.append(f'<a href="{html.escape(label)}">{html.escape(label)}</a>')
    artifact_links = "".join(links)
    reason = why_ended(snapshot)
    goal = snapshot.run.goal_text
    status = terminal_reason(snapshot)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoevolve run {html.escape(snapshot.run.id)}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f4f5f7;
  --surface: #ffffff;
  --surface-soft: #f8f9fa;
  --text: #20242b;
  --muted: #68707b;
  --border: #dfe2e6;
  --blue: #28689b;
  --accent: {ACCENT};
  --shadow: 0 16px 44px rgba(28, 36, 46, 0.08);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #111418;
    --surface: #191d22;
    --surface-soft: #20252b;
    --text: #edf0f3;
    --muted: #9ba3ad;
    --border: #30363e;
    --blue: #77aeda;
    --shadow: 0 16px 44px rgba(0, 0, 0, 0.28);
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}}
main {{ max-width: 1480px; margin: 0 auto; padding: 38px 28px 60px; }}
header {{ display: grid; gap: 18px; margin-bottom: 24px; }}
.eyebrow {{
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}
.title-row {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
h1 {{ margin: 0; font-size: clamp(1.8rem, 4vw, 3.2rem); letter-spacing: -0.04em; }}
h2 {{ margin: 0 0 16px; font-size: 1rem; letter-spacing: -0.01em; }}
p {{ margin: 0; }}
.status {{
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 5px 11px;
  color: var(--muted);
  background: var(--surface);
  font-size: 0.78rem;
}}
.goal {{ max-width: 920px; font-size: 1.16rem; }}
.reason {{ max-width: 1020px; color: var(--muted); }}
.grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(320px, 0.75fr);
  gap: 20px;
}}
.stack {{ display: grid; gap: 20px; align-content: start; }}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 20px;
  min-width: 0;
}}
.panel {{ overflow: hidden; }}
.panel svg {{ display: block; width: 100%; height: auto; min-height: 260px; }}
.contract {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  margin: 0;
}}
.contract div {{ background: var(--surface-soft); border-radius: 10px; padding: 10px 12px; }}
.contract dt {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; }}
.contract dd {{ margin: 4px 0 0; font-size: 0.9rem; overflow-wrap: anywhere; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.86rem;
}}
th, td {{ padding: 9px 7px; border-bottom: 1px solid var(--border); text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
.muted {{ color: var(--muted); }}
.discoveries {{ list-style: none; display: grid; gap: 11px; padding: 0; margin: 0; }}
.discoveries li {{ display: grid; gap: 3px; }}
.discoveries small {{ color: var(--muted); }}
.artifacts {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.artifacts a {{
  color: var(--blue);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 9px;
  text-decoration: none;
  font-size: 0.82rem;
}}
.artifacts a:hover {{ border-color: var(--blue); }}
.tooltip {{
  position: fixed;
  z-index: 20;
  pointer-events: none;
  display: none;
  max-width: 320px;
  padding: 9px 11px;
  border-radius: 8px;
  background: #12161b;
  color: #f7f8fa;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  font-size: 0.76rem;
  white-space: pre-line;
}}
@media (max-width: 900px) {{
  main {{ padding: 24px 15px 44px; }}
  .grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">autoevolve run</div>
    <div class="title-row">
      <h1>{html.escape(snapshot.run.id)}</h1>
      <span class="status">{html.escape(status)}</span>
    </div>
    <p class="goal">{html.escape(goal)}</p>
    <p class="reason">{html.escape(reason)}</p>
    <dl class="contract">{contract_html}</dl>
  </header>
  <div class="grid">
    <section class="stack">
      <article class="card panel"><h2>Lineage</h2>{lineage_svg}</article>
      <article class="card panel"><h2>Best-score curve</h2>{curve_svg}</article>
    </section>
    <aside class="stack">
      <section class="card">
        <h2>Operator and bandit stats</h2>
        <table><thead><tr>
          <th>operator</th><th>pulls</th><th>improved</th><th>mean gain</th>
        </tr></thead>
        <tbody>{operator_rows}</tbody></table>
      </section>
      <section class="card">
        <h2>Islands</h2>
        <table><thead><tr><th>island</th><th>evals</th><th>best</th><th>score</th></tr></thead>
        <tbody>{island_rows}</tbody></table>
      </section>
      <section class="card">
        <h2>Discoveries for {html.escape(snapshot.run.domain)}</h2>
        <ul class="discoveries">{discovery_items}</ul>
      </section>
      <section class="card">
        <h2>Artifacts</h2>
        <div class="artifacts">{artifact_links}</div>
      </section>
    </aside>
  </div>
</main>
<div class="tooltip" id="node-tooltip"></div>
<script>
(() => {{
  const tooltip = document.getElementById("node-tooltip");
  for (const node of document.querySelectorAll("[data-tooltip]")) {{
    node.addEventListener("pointerenter", () => {{
      tooltip.textContent = node.dataset.tooltip;
      tooltip.style.display = "block";
    }});
    node.addEventListener("pointermove", event => {{
      tooltip.style.left = `${{event.clientX + 14}}px`;
      tooltip.style.top = `${{event.clientY + 14}}px`;
    }});
    node.addEventListener("pointerleave", () => {{ tooltip.style.display = "none"; }});
  }}
}})();
</script>
</body>
</html>
"""


def _lineage_svg(
    snapshot: Snapshot,
    layout: dict[str, tuple[float, float]],
) -> str:
    width = 920
    height = 540
    padding_x = 54
    padding_y = 42
    if not layout:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Empty lineage"><text x="{width / 2}" y="{height / 2}" '
            'text-anchor="middle" fill="currentColor">No programs recorded.</text></svg>'
        )
    x_values = [point[0] for point in layout.values()]
    y_values = [point[1] for point in layout.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        projected_x = padding_x + (x - x_min) / x_span * (width - 2 * padding_x)
        projected_y = padding_y + (y - y_min) / y_span * (height - 2 * padding_y)
        return projected_x, projected_y

    path = elite_path(snapshot)
    path_edges = set(zip(path[1:], path[:-1], strict=False))
    scores = [
        _objective_score(snapshot, snapshot.score(program.id))
        for program in snapshot.programs
        if snapshot.is_scored(program.id)
    ]
    measured = [score for score in scores if score is not None]
    low = min(measured, default=0.0)
    high = max(measured, default=1.0)
    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Lineage graph for {html.escape(snapshot.run.id)}">'
    ]
    for island in sorted({program.island for program in snapshot.programs}):
        lane_x, _ = project((island * 2.0, y_min))
        pieces.append(
            f'<line x1="{lane_x:.2f}" y1="20" x2="{lane_x:.2f}" y2="{height - 18}" '
            'stroke="var(--border)" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{lane_x:.2f}" y="16" text-anchor="middle" fill="var(--muted)" '
            f'font-size="11">island {island}</text>'
        )
    for child_id, parent_id, kind in snapshot.edges:
        if child_id not in layout or parent_id not in layout:
            continue
        child_x, child_y = project(layout[child_id])
        parent_x, parent_y = project(layout[parent_id])
        highlighted = (child_id, parent_id) in path_edges
        dash = ' stroke-dasharray="5 4"' if kind == "migration" else ""
        pieces.append(
            f'<line x1="{parent_x:.2f}" y1="{parent_y:.2f}" '
            f'x2="{child_x:.2f}" y2="{child_y:.2f}" '
            f'stroke="{ACCENT if highlighted else EDGE_COLOR}" '
            f'stroke-width="{2.8 if highlighted else 1.0}"{dash}/>'
        )
    diff_counts = _diff_summary_counts(snapshot)
    for program in snapshot.programs:
        x, y = project(layout[program.id])
        scored = snapshot.is_scored(program.id)
        score = snapshot.score(program.id)
        color = (
            _svg_fitness_color(_objective_score(snapshot, score), low, high)
            if scored
            else "none"
        )
        stroke = ACCENT if program.id in path else FAILED_COLOR
        score_text = ", ".join(
            f"{name}={format_number(value)}"
            for name, value in sorted(snapshot.scores.get(program.id, {}).items())
        ) or "none"
        tooltip = (
            f"program: {program.id}\noperator: {program.operator}\nisland: {program.island}"
            f"\nscores: {score_text}\ndiff summary count: {diff_counts.get(program.id, 0)}"
        )
        title = html.escape(tooltip)
        pieces.append(
            f'<g data-tooltip="{html.escape(tooltip, quote=True)}" tabindex="0">'
            f"<title>{title}</title>"
            f'{_node_shape(program.operator, x, y, color, stroke, program.id in path)}'
            "</g>"
        )
    pieces.append(_operator_legend(width, height))
    pieces.append("</svg>")
    return "".join(pieces)


def _curve_svg(snapshot: Snapshot) -> str:
    width = 920
    height = 300
    left = 58
    right = 24
    top = 22
    bottom = 42
    curve = snapshot.curve()
    if not curve:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Empty score curve">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            'fill="currentColor">No measured scores recorded.</text></svg>'
        )
    x_values = [point.eval_idx for point in curve]
    y_values = [point.value for point in curve]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if y_max <= y_min:
        y_min -= 0.5
        y_max += 0.5

    def point(eval_idx: int, value: float) -> tuple[float, float]:
        x_span = max(1, x_max - x_min)
        x = left + (eval_idx - x_min) / x_span * (width - left - right)
        y = top + (y_max - value) / (y_max - y_min) * (height - top - bottom)
        return x, y

    coordinates = [point(item.eval_idx, item.value) for item in curve]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    grid = []
    for index in range(5):
        y = top + index * (height - top - bottom) / 4
        value = y_max - index * (y_max - y_min) / 4
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'fill="var(--muted)" font-size="10">{html.escape(format_number(value))}</text>'
        )
    final_x, final_y = coordinates[-1]
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Best {html.escape(snapshot.metric)} by evaluation">'
        f'{"".join(grid)}'
        f'<polyline points="{polyline}" fill="none" stroke="var(--blue)" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{final_x:.2f}" cy="{final_y:.2f}" r="4" fill="var(--blue)"/>'
        f'<text x="{left}" y="{height - 12}" fill="var(--muted)" '
        'font-size="10">0 evaluations</text>'
        f'<text x="{width - right}" y="{height - 12}" text-anchor="end" '
        f'fill="var(--muted)" font-size="10">{x_max} evaluations</text>'
        "</svg>"
    )


def _node_shape(
    operator: str,
    x: float,
    y: float,
    fill: str,
    stroke: str,
    highlighted: bool,
) -> str:
    width = 2.2 if highlighted else 1.3
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"'
    if operator == "rewrite":
        return f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" {common}/>'
    if operator == "agentic":
        points = f"{x:.2f},{y - 6:.2f} {x - 5.5:.2f},{y + 5:.2f} {x + 5.5:.2f},{y + 5:.2f}"
        return f'<polygon points="{points}" {common}/>'
    if operator == "crossover":
        points = f"{x:.2f},{y - 6:.2f} {x - 6:.2f},{y:.2f} {x:.2f},{y + 6:.2f} {x + 6:.2f},{y:.2f}"
        return f'<polygon points="{points}" {common}/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" {common}/>'


def _operator_legend(width: int, height: int) -> str:
    labels = (
        ("diff", "circle"),
        ("rewrite", "square"),
        ("agentic", "triangle"),
        ("crossover", "diamond"),
    )
    items = []
    x = 18.0
    y = float(height - 14)
    for operator, label in labels:
        items.append(_node_shape(operator, x, y - 3, "var(--blue)", "var(--blue)", False))
        items.append(
            f'<text x="{x + 10:.2f}" y="{y:.2f}" fill="var(--muted)" '
            f'font-size="10">{label}</text>'
        )
        x += 78
    return "".join(items)


def _diff_summary_counts(snapshot: Snapshot) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in snapshot.events:
        program_id = event_program_id(event)
        if program_id is None:
            continue
        payload = event.payload
        for key in ("diff_summary_count", "files_changed", "changed_files"):
            value: Any = payload.get(key)
            if isinstance(value, int):
                counts[program_id] = value
                break
            if isinstance(value, list | dict):
                counts[program_id] = len(value)
                break
    return counts


def _svg_fitness_color(value: float | None, low: float, high: float) -> str:
    if value is None:
        return FAILED_COLOR
    ratio = 0.65 if high <= low else (value - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    start = (218, 232, 244)
    end = (35, 99, 154)
    red = round(start[0] + ratio * (end[0] - start[0]))
    green = round(start[1] + ratio * (end[1] - start[1]))
    blue = round(start[2] + ratio * (end[2] - start[2]))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _objective_score(snapshot: Snapshot, value: float | None) -> float | None:
    if value is None:
        return None
    return value if snapshot.maximize else -value
