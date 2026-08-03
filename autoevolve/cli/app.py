"""Typer command-line entrypoint for autoevolve."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import typer

from autoevolve.cli._data import RunNotFoundError, artifact_dir, artifact_paths, home_from_env
from autoevolve.core.types import Budget

app = typer.Typer(
    name="autoevolve",
    help="Agent-native evolutionary optimization.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

OPERATORS = ("diff", "rewrite", "agentic", "crossover")


@app.command("init")
def init_command(
    name: Annotated[Path, typer.Argument(help="Evaluator folder to create.")],
) -> None:
    """Scaffold a documented evaluator with a runnable minimal example."""

    target = name.expanduser()
    if target.exists():
        if not target.is_dir():
            _abort(f"Refusing to overwrite a file: {target}")
        if any(target.iterdir()):
            _abort(f"Refusing to overwrite non-empty evaluator folder: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "baseline").mkdir(exist_ok=True)
    (target / "fixtures").mkdir(exist_ok=True)
    (target / "spec.md").write_text(_SPEC_TEMPLATE, encoding="utf-8")
    (target / "evaluate.py").write_text(_EVALUATE_TEMPLATE, encoding="utf-8")
    (target / "baseline" / "candidate.py").write_text(_BASELINE_TEMPLATE, encoding="utf-8")
    (target / "fixtures" / "cases.json").write_text(
        json.dumps({"cases": [{"input": 2, "expected": 4}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Created evaluator template at {target.resolve()}")
    typer.echo("Next steps:")
    typer.echo("  1. Define the metric, gate, target, hardware, and fixture provenance in spec.md.")
    typer.echo("  2. Replace the worked evaluate.py example with measurements for your domain.")
    typer.echo("  3. Put the seed implementation in baseline/ and correctness data in fixtures/.")
    typer.echo(f"  4. Run: autoevolve run --evaluator {target}")


@app.command("run")
def run_command(
    evaluator: Annotated[
        Path,
        typer.Option(
            "--evaluator",
            help="Evaluator directory containing spec.md and evaluate.py.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    goal: Annotated[str | None, typer.Option("--goal", help="Goal text to lock.")] = None,
    budget_evals: Annotated[
        int | None,
        typer.Option("--budget-evals", min=1, help="Maximum child evaluations."),
    ] = None,
    wall_clock_s: Annotated[
        float | None,
        typer.Option("--wall-clock-s", min=0.001, help="Maximum elapsed seconds."),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, help="Number of islands.")] = 4,
    parallel: Annotated[
        int,
        typer.Option(
            "--parallel",
            min=1,
            help="Worker threads driving this run. Each takes its own island.",
        ),
    ] = 1,
    seed: Annotated[int | None, typer.Option("--seed", help="Replay seed.")] = None,
    target: Annotated[
        float | None,
        typer.Option("--target", help="Metric value that ends the run as target_hit."),
    ] = None,
    operators: Annotated[
        str,
        typer.Option("--operators", help="Comma-separated operator allowlist."),
    ] = "diff,rewrite,agentic,crossover",
) -> None:
    """Open a run and drive the core worker loop in this process."""

    budget = Budget(max_evals=budget_evals, wall_clock_s=wall_clock_s)
    if not budget.is_bounded():
        _abort("A run requires at least one budget bound: --budget-evals or --wall-clock-s.")
    operator_names = _parse_operators(operators)
    from autoevolve.core.engine import Engine

    home = home_from_env()
    engine = Engine(home=home)
    goal_text = goal or f"Optimize {evaluator.name}"
    try:
        opened = engine.open_run(
            goal_text=goal_text,
            evaluator_ref=str(evaluator),
            budget=budget,
            workers=workers,
            seed=seed,
            target=target,
        )
        run_id = str(opened["run_id"])
        if opened.get("status") == "infeasible" or bool(opened.get("infeasible")):
            typer.echo(f"Opened {run_id}; ceiling analysis closed it as infeasible.")
        else:
            get_operator = _build_get_operator(operator_names, evaluator)
            summaries = _drive_workers(engine, run_id, get_operator, parallel, operator_names)
            _print_worker_summaries(summaries)
    except (ValueError, RuntimeError) as exc:
        _abort(str(exc))
    _finish_and_print(engine, home, run_id)


@app.command("watch")
def watch_command(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    refresh: Annotated[
        float,
        typer.Option("--refresh", min=0.05, max=60.0, help="Refresh interval in seconds."),
    ] = 1.0,
    render_live: Annotated[
        bool,
        typer.Option("--render-live", help="Regenerate latest.png on a timer."),
    ] = False,
) -> None:
    """Watch a run through the read-only Rich terminal dashboard."""

    from autoevolve.cli.tui import watch_run

    home = home_from_env()
    try:
        watch_run(
            home,
            run_id,
            refresh=refresh,
            render_live=render_live,
            out_dir=artifact_dir(run_id),
        )
    except RunNotFoundError as exc:
        _abort(str(exc), code=1)


@app.command("join")
def join_command(
    run_id: Annotated[str, typer.Argument(help="Existing run identifier.")],
    island: Annotated[
        int | None,
        typer.Option("--island", min=0, help="Override the assigned island."),
    ] = None,
) -> None:
    """Join the current terminal to an existing run as a worker."""

    from autoevolve.core.engine import Engine
    from autoevolve.core.loop import run_worker_loop

    home = home_from_env()
    engine = Engine(home=home)
    try:
        joined = engine.join_run(run_id, runtime="cli")
        assigned = int(joined["island"])
        selected = assigned if island is None else island
        typer.echo(f"Joined {run_id} on island {selected}.")
        from autoevolve.cli._data import load_snapshot

        evaluator_ref = load_snapshot(home, run_id).run.evaluator_ref
        evaluator_dir = Path(evaluator_ref) if evaluator_ref else None
        run_worker_loop(
            engine,
            run_id,
            _build_get_operator(None, evaluator_dir),
            island=selected,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        _abort(str(exc), code=1)
    _finish_and_print(engine, home, run_id)


@app.command("report")
def report_command(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    out: Annotated[Path | None, typer.Option("--out", help="Markdown output path.")] = None,
) -> None:
    """Write report.md for a run."""

    from autoevolve.cli.report import report

    output = out.expanduser() if out else artifact_dir(run_id) / "report.md"
    try:
        result = report(home_from_env(), run_id, output)
    except RunNotFoundError as exc:
        _abort(str(exc), code=1)
    typer.echo(result)


@app.command("render")
def render_command(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    out: Annotated[Path | None, typer.Option("--out", help="Artifact output directory.")] = None,
    live: Annotated[
        bool,
        typer.Option("--live", help="Also write latest.png from the current state."),
    ] = False,
) -> None:
    """Render the animation, poster, and dashboard for a run."""

    from autoevolve.cli.render import render_all

    output = out.expanduser() if out else artifact_dir(run_id)
    try:
        results = render_all(home_from_env(), run_id, output, live=live)
    except RunNotFoundError as exc:
        _abort(str(exc), code=1)
    for key, path in results.items():
        typer.echo(f"{key}: {path if path is not None else 'not generated'}")


@app.command("serve")
def serve_command(
    http: Annotated[bool, typer.Option("--http", help="Serve HTTP instead of stdio.")] = False,
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8747,
) -> None:
    """Start the MCP server over stdio or HTTP."""

    server = importlib.import_module("autoevolve.mcp.server")

    if http:
        server.serve_http(port=port)
    else:
        server.serve_stdio()


def main() -> None:
    """Invoke the Typer application for the console-script entrypoint."""

    app()


def _parse_operators(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(names) - set(OPERATORS))
    if not names:
        _abort("At least one operator is required.")
    if invalid:
        _abort(f"Unknown operators: {', '.join(invalid)}. Choose from {', '.join(OPERATORS)}.")
    return list(dict.fromkeys(names))


def _drive_workers(
    engine: Any,
    run_id: str,
    get_operator: Callable[[str], object],
    parallel: int,
    operator_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run one or more worker threads against a single open run.

    A cycle spends nearly all of its wall clock waiting on a model call, so
    workers overlap almost perfectly and throughput scales close to linearly
    with the thread count. Each thread joins separately and gets its own
    island, which is what keeps their populations distinct. Writes are
    serialized by the store's write lock, so the engine is shared safely.
    """

    from autoevolve.core.loop import run_worker_loop

    def work(index: int) -> dict[str, Any]:
        joined = engine.join_run(run_id, runtime=f"cli-{index}")
        island = int(joined["island"])
        typer.echo(f"worker {index} joined island {island}.")
        return run_worker_loop(
            engine,
            run_id,
            get_operator,
            island=island,
            operators=tuple(operator_names) if operator_names else None,
        )

    if parallel == 1:
        return [work(0)]

    summaries: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(work, index) for index in range(parallel)]
        for future in as_completed(futures):
            try:
                summaries.append(future.result())
            except BaseException as exc:  # noqa: BLE001 - reported after the join
                errors.append(exc)
    if errors and not summaries:
        raise errors[0]
    for exc in errors:
        typer.echo(f"worker failed: {exc}", err=True)
    return summaries


def _distill_discoveries(home: Path, run_id: str) -> None:
    """Turn this run's best lineage into discoveries later runs can sample.

    This is the mechanism that makes knowledge compound across runs. It is
    best effort: a run that produced real results must never be reported as
    failed because distillation could not reach a model.
    """

    try:
        from autoevolve.mutate.distiller import distill_run
        from autoevolve.mutate.models import resolve_endpoint

        endpoint = resolve_endpoint("strong") or resolve_endpoint("cheap")
        if endpoint is None:
            return
        found = distill_run(home, run_id, endpoint)
    except Exception as exc:  # noqa: BLE001 - never fail a finished run on this
        typer.echo(f"discovery distillation skipped: {exc}", err=True)
        return
    if found:
        typer.echo(f"distilled {len(found)} discoveries for future runs.")


def _print_worker_summaries(summaries: list[Any]) -> None:
    """Report what the workers did. A summary line never fails a finished run."""

    usable = [item for item in summaries if isinstance(item, dict)]
    total_submissions = sum(int(item.get("submissions", 0)) for item in usable)
    total_skips = sum(int(item.get("skips", 0)) for item in usable)
    typer.echo(
        f"{len(summaries)} worker(s) finished: {total_submissions} submissions, "
        f"{total_skips} skipped cycles."
    )


def _build_local_evaluator(evaluator_dir: Path | None) -> Callable[[dict[str, str]], Any] | None:
    """Patchable seam over the shared composition helper."""

    from autoevolve.mutate.compose import build_local_evaluator

    return build_local_evaluator(evaluator_dir)


def _build_get_operator(
    operator_names: list[str] | None,
    evaluator_dir: Path | None,
) -> Callable[[str], object]:
    """Compose mutate operators with their runtime services for the core loop."""

    from autoevolve.mutate.compose import build_get_operator

    return build_get_operator(operator_names, _build_local_evaluator(evaluator_dir))


def _finish_and_print(engine: Any, home: Path, run_id: str) -> None:
    _distill_discoveries(home, run_id)
    out_dir = artifact_dir(run_id)
    paths = artifact_paths(out_dir)
    try:
        from autoevolve.cli._data import load_snapshot, why_ended
        from autoevolve.cli.render import render_all
        from autoevolve.cli.report import report

        snapshot = load_snapshot(home, run_id)
        render_all(home, run_id, out_dir)
        report(home, run_id, paths["report"])
        typer.echo(why_ended(snapshot))
    except RunNotFoundError:
        status = engine.run_status(run_id)
        typer.echo(_status_paragraph(status))
    typer.echo(f"dashboard: {paths['dashboard']}")
    typer.echo(f"gif: {paths['gif']}")
    typer.echo(f"mp4: {paths['mp4'] if paths['mp4'].exists() else 'not generated'}")
    typer.echo(f"poster svg: {paths['poster_svg']}")
    typer.echo(f"poster: {paths['poster_png']}")
    typer.echo(f"report: {paths['report']}")


def _status_paragraph(status: dict[str, Any]) -> str:
    reason = str(status.get("status", "closed"))
    if reason == "target_hit":
        return "The run ended because its locked target was reached."
    if reason in {"budget", "budget_exhausted"}:
        return "The run ended because its configured budget was exhausted."
    if reason == "plateau":
        return "The run ended because it reached its configured plateau limit."
    if reason == "infeasible":
        return "The run ended because ceiling analysis proved the locked target infeasible."
    return f"The run ended with status {reason}."


def _abort(message: str, *, code: int = 2) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


_SPEC_TEMPLATE = """# Evaluator specification

## Goal

Describe the optimization goal in plain language.

## Metrics and units

- `score`: higher is better; replace this with the measured metric and its unit.
- `correct`: boolean correctness gate represented as 1.0 or 0.0.

## Correctness gate

State exactly what must remain correct before any score counts.

## Target semantics

State whether the metric is maximized or minimized and define the target value.

## Hardware needs

CPU only for this example. Declare accelerators and a CPU mock path when required.

## Fixture provenance

The example fixture is hand-authored for the square function. Replace it and record how
the production fixtures were created.
"""


_EVALUATE_TEMPLATE = '''"""Worked minimal evaluator for the scaffolded square-function baseline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from autoevolve.eval.contract import EvalError, StageSpec


STAGES: list[StageSpec] = [StageSpec(name="correctness", timeout_s=5.0)]
GATE = "correct"


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Measure fixture correctness and reward a compact candidate source file."""

    del stage
    module_path = candidate_dir / "candidate.py"
    spec = importlib.util.spec_from_file_location("candidate", module_path)
    if spec is None or spec.loader is None:
        raise EvalError("candidate.py could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture_path = Path(__file__).parent / "fixtures" / "cases.json"
    payload: dict[str, Any] = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if module.solve(case["input"]) != case["expected"]:
            raise EvalError(f"wrong result for input {case['input']}")
    source_bytes = len(module_path.read_bytes())
    return {GATE: 1.0, "score": 1.0 / max(1, source_bytes)}


def ceiling() -> dict[str, Any] | None:
    """Return no theoretical ceiling for this worked example."""

    return None
'''


_BASELINE_TEMPLATE = '''"""Seed candidate for the worked evaluator."""


def solve(value: int) -> int:
    """Return the square of an integer."""

    return value * value
'''


try:
    from autoevolve.cli import campaign
except ImportError:
    campaign = None
else:
    campaign.register(app)
