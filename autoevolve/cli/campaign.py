"""Research campaign pack discovery, execution, and database reports."""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from autoevolve.cli._data import (
    Snapshot,
    artifact_dir,
    format_number,
    home_from_env,
    load_snapshot,
    terminal_reason,
)
from autoevolve.cli.claims_lint import ClaimViolation, scan_repository
from autoevolve.core.engine import Engine
from autoevolve.core.loop import run_worker_loop
from autoevolve.core.types import Budget, EvalOutcome

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS_ROOT = REPO_ROOT / "campaigns"
_BUDGET_KEYS = frozenset({"max_evals", "wall_clock_s", "max_cost_usd"})
_CONTRACT_OVERRIDES: dict[str, tuple[str, bool]] = {
    "arch-search": ("val_loss", False),
    "algorithm-frontier": ("bins_used", False),
    "equation-discovery": ("fitness", True),
}
_REGISTERED_APPS: set[int] = set()

campaign_app = typer.Typer(
    name="campaign",
    help="Run and report reproducible research campaign packs.",
    no_args_is_help=True,
)


class CampaignError(ValueError):
    """Raised when a campaign pack or command request is invalid."""


@dataclass(frozen=True)
class CampaignCell:
    """One independently tagged campaign target."""

    key: str
    env: dict[str, str]
    target: float | None


@dataclass(frozen=True)
class CampaignConfig:
    """Validated machine configuration for one campaign pack."""

    pack_dir: Path
    name: str
    domain: str
    evaluator: str
    cells: tuple[CampaignCell, ...]
    proxy_budget: dict[str, int | float | None]
    full_budget: dict[str, int | float | None]
    ladder: tuple[str, ...]
    replicate_seeds: int

    @property
    def evaluator_path(self) -> Path:
        return (self.pack_dir / self.evaluator).resolve()

    def select_cells(self, key: str | None) -> tuple[CampaignCell, ...]:
        """Return every cell or one exact requested cell."""

        if key is None:
            return self.cells
        selected = tuple(cell for cell in self.cells if cell.key == key)
        if not selected:
            choices = ", ".join(cell.key for cell in self.cells)
            raise CampaignError(
                f"campaign {self.name!r} has no cell {key!r}; choose from {choices}"
            )
        return selected

    def budget(self, *, full: bool) -> Budget:
        """Return the selected bounded core budget."""

        values = self.full_budget if full else self.proxy_budget
        return Budget(
            max_evals=_optional_int(values.get("max_evals")),
            wall_clock_s=_optional_float(values.get("wall_clock_s")),
            max_cost_usd=_optional_float(values.get("max_cost_usd")),
        )


@dataclass(frozen=True)
class CampaignRunResult:
    """Recorded result from one cell execution."""

    run_id: str
    cell: str
    best_fitness: float | None
    end_cause: str


@dataclass(frozen=True)
class _RunFact:
    run_id: str
    seed: int
    completed: bool
    metric: str
    best_value: float | None
    best_fitness: float | None
    improved: bool
    r2_heldout: float | None


@dataclass(frozen=True)
class _ConfiguredEvaluator:
    base: Any
    domain: str
    metric: str | None
    maximize: bool
    target: float | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def ceiling(self) -> dict[str, Any] | None:
        return self.base.ceiling()


@dataclass(frozen=True)
class _OperatorAdapter:
    operator: Any
    evaluator_dir: Path

    def propose(self, bundle: Any, context: Any) -> Any:
        from autoevolve.mutate.base import OperatorContext
        from autoevolve.mutate.models import resolve_endpoint

        enriched = OperatorContext(
            contract=context.contract,
            rng=context.rng,
            endpoint_cheap=resolve_endpoint("cheap"),
            endpoint_strong=resolve_endpoint("strong"),
            evaluate_locally=self._evaluate_locally,
            workdir=context.workdir,
        )
        return self.operator.propose(bundle, enriched)

    def _evaluate_locally(self, files: dict[str, str]) -> EvalOutcome:
        from autoevolve.eval.cascade import run_cascade
        from autoevolve.eval.contract import load_evaluator

        evaluator = load_evaluator(self.evaluator_dir)
        with tempfile.TemporaryDirectory(prefix="autoevolve-campaign-local-") as raw:
            candidate_dir = Path(raw)
            for relative, content in files.items():
                target = _safe_candidate_path(candidate_dir, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return run_cascade(evaluator, candidate_dir)


def register(app: typer.Typer) -> None:
    """Register the campaign command group on the root Typer application."""

    identity = id(app)
    if identity in _REGISTERED_APPS:
        return
    app.add_typer(campaign_app, name="campaign")
    _REGISTERED_APPS.add(identity)


def discover_campaigns(root: Path = CAMPAIGNS_ROOT) -> tuple[CampaignConfig, ...]:
    """Discover and validate every campaign pack under a root directory."""

    if not root.is_dir():
        return ()
    configs = [
        load_campaign(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "campaign.json").is_file()
    ]
    names = [config.name for config in configs]
    if len(names) != len(set(names)):
        raise CampaignError("campaign names must be unique")
    return tuple(configs)


def load_campaign(pack_dir: Path) -> CampaignConfig:
    """Parse and validate one campaign.json with clear field diagnostics."""

    config_path = pack_dir / "campaign.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CampaignError(f"missing campaign config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(
            f"invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(raw, dict):
        raise CampaignError(f"{config_path} must contain one JSON object")

    required = {
        "name",
        "domain",
        "evaluator",
        "cells",
        "proxy_budget",
        "full_budget",
        "ladder",
        "replicate_seeds",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise CampaignError(f"{config_path} is missing required keys: {', '.join(missing)}")
    unknown = sorted(set(raw) - required)
    if unknown:
        raise CampaignError(f"{config_path} has unknown keys: {', '.join(unknown)}")

    name = _nonempty_string(raw["name"], "name", config_path)
    if name != pack_dir.name:
        raise CampaignError(
            f"{config_path} name {name!r} must match directory {pack_dir.name!r}"
        )
    domain = _nonempty_string(raw["domain"], "domain", config_path)
    evaluator = _nonempty_string(raw["evaluator"], "evaluator", config_path)
    cells = _parse_cells(raw["cells"], config_path)
    proxy_budget = _parse_budget(raw["proxy_budget"], "proxy_budget", config_path)
    full_budget = _parse_budget(raw["full_budget"], "full_budget", config_path)
    ladder = _parse_ladder(raw["ladder"], config_path)
    replicate_seeds = raw["replicate_seeds"]
    if (
        isinstance(replicate_seeds, bool)
        or not isinstance(replicate_seeds, int)
        or replicate_seeds <= 0
    ):
        raise CampaignError(f"{config_path} replicate_seeds must be a positive integer")

    config = CampaignConfig(
        pack_dir=pack_dir.resolve(),
        name=name,
        domain=domain,
        evaluator=evaluator,
        cells=cells,
        proxy_budget=proxy_budget,
        full_budget=full_budget,
        ladder=ladder,
        replicate_seeds=replicate_seeds,
    )
    if not config.evaluator_path.is_dir():
        raise CampaignError(
            f"{config_path} evaluator directory does not exist: {config.evaluator_path}"
        )
    return config


def find_campaign(name: str, root: Path = CAMPAIGNS_ROOT) -> CampaignConfig:
    """Return one named discovered pack or a clear selection error."""

    configs = discover_campaigns(root)
    for config in configs:
        if config.name == name:
            return config
    choices = ", ".join(config.name for config in configs) or "none"
    raise CampaignError(f"unknown campaign {name!r}; available campaigns: {choices}")


def execute_campaign(
    config: CampaignConfig,
    *,
    cell_key: str | None,
    full: bool,
    seed: int | None,
    home: Path,
) -> list[CampaignRunResult]:
    """Open, drive, render, report, and log one run per selected cell."""

    results: list[CampaignRunResult] = []
    for cell in config.select_cells(cell_key):
        engine = _build_engine(home, config, cell)
        budget = config.budget(full=full)
        budget_values = config.full_budget if full else config.proxy_budget
        run_id: str | None = None
        failure: Exception | None = None
        with _cell_environment(cell.env):
            opened = engine.open_run(
                goal_text=f"campaign:{config.name}:{cell.key}",
                evaluator_ref=str(config.evaluator_path),
                budget=budget,
                workers=4,
                seed=seed,
            )
            run_id = str(opened["run_id"])
            try:
                if str(opened.get("status", "open")) == "open":
                    run_worker_loop(
                        engine,
                        run_id,
                        _operator_factory(config.evaluator_path),
                    )
                _write_run_artifacts(home, run_id)
            except Exception as exc:
                failure = exc

        assert run_id is not None
        status = _run_end_cause(engine, run_id, failure)
        best_fitness = _best_fitness(engine, run_id)
        _append_log_block(
            config.pack_dir / "log.md",
            run_id=run_id,
            cell=cell.key,
            budget=budget_values,
            best_fitness=best_fitness,
            end_cause=status,
        )
        result = CampaignRunResult(run_id, cell.key, best_fitness, status)
        results.append(result)
        if failure is not None:
            raise CampaignError(f"campaign run {run_id} failed: {failure}") from failure
    return results


def build_campaign_report(
    home: Path,
    config: CampaignConfig,
    *,
    claims_root: Path | None = REPO_ROOT,
) -> str:
    """Reconstruct one campaign report from normative database facts only."""

    snapshots = _campaign_snapshots(home, config.name)
    facts_by_cell: dict[str, list[_RunFact]] = {cell.key: [] for cell in config.cells}
    prefix = f"campaign:{config.name}:"
    for snapshot in snapshots:
        cell = snapshot.run.goal_text.removeprefix(prefix)
        if cell in facts_by_cell:
            facts_by_cell[cell].append(_run_fact(snapshot))

    rows: list[str] = []
    for cell in config.cells:
        facts = facts_by_cell[cell.key]
        best = _best_fact(facts)
        ladder, classification = _labels(config, facts, best)
        run_id = best.run_id if best is not None else "not run"
        metric = best.metric if best is not None else "fitness"
        value = format_number(best.best_value if best is not None else None)
        rows.append(
            f"| {cell.key} | `{run_id}` | `{metric}` {value} | "
            f"{classification} | {ladder} |"
        )

    claims = scan_repository(claims_root) if claims_root is not None else []
    claims_section = _claims_report(claims, claims_root)
    return (
        f"# Campaign report: {config.name}\n\n"
        f"Domain: `{config.domain}`\n\n"
        "| cell | best run | best measured result | honesty label | ladder position |\n"
        "|---|---|---:|---|---|\n"
        + "\n".join(rows)
        + "\n\nScaled validation requires an explicit scaled validation run. "
        "It is not run or claimed automatically.\n\n"
        + claims_section
    )


@campaign_app.command("list")
def list_command() -> None:
    """List every discovered campaign pack."""

    try:
        configs = discover_campaigns()
    except CampaignError as exc:
        _abort(str(exc))
    if not configs:
        typer.echo("No campaign packs found.")
        return
    for config in configs:
        cells = ",".join(cell.key for cell in config.cells)
        ladder = " -> ".join(config.ladder)
        typer.echo(f"{config.name}\t{config.domain}\t{cells}\t{ladder}")


@campaign_app.command("run")
def run_command(
    name: Annotated[str, typer.Argument(help="Campaign pack name.")],
    cell: Annotated[
        str | None,
        typer.Option("--cell", help="Run one campaign cell by key."),
    ] = None,
    proxy: Annotated[
        bool,
        typer.Option("--proxy", help="Use the default proxy budget."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Use the opt-in full budget."),
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Replay seed.")] = None,
) -> None:
    """Run selected campaign cells through the local worker loop."""

    if proxy and full:
        _abort("Choose only one of --proxy or --full.")
    try:
        config = find_campaign(name)
        results = execute_campaign(
            config,
            cell_key=cell,
            full=full,
            seed=seed,
            home=home_from_env(),
        )
    except (CampaignError, KeyError, OSError, RuntimeError) as exc:
        _abort(str(exc), code=1)
    for result in results:
        typer.echo(
            f"{result.run_id}\t{result.cell}\t"
            f"best={format_number(result.best_fitness)}\tend={result.end_cause}"
        )


@campaign_app.command("report")
def report_command(
    name: Annotated[str, typer.Argument(help="Campaign pack name.")],
) -> None:
    """Print a database-derived campaign ladder report."""

    try:
        config = find_campaign(name)
        content = build_campaign_report(home_from_env(), config)
    except (CampaignError, OSError, sqlite3.Error) as exc:
        _abort(str(exc), code=1)
    typer.echo(content)


def _build_engine(home: Path, config: CampaignConfig, cell: CampaignCell) -> Any:
    loader = _configured_loader(config, cell)
    try:
        parameters = inspect.signature(Engine).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "evaluator_loader" in parameters:
        return Engine(home=home, evaluator_loader=loader)
    return Engine(home=home)


def _configured_loader(
    config: CampaignConfig,
    cell: CampaignCell,
) -> Callable[[Path], Any]:
    def load(path: Path) -> Any:
        from autoevolve.eval.contract import load_evaluator

        base = load_evaluator(path)
        override = _CONTRACT_OVERRIDES.get(config.name)
        if override is not None:
            metric, maximize = override
        else:
            metric = getattr(base, "metric", None)
            declared = getattr(base, "maximize", None)
            maximize = declared if declared is not None else True
        return _ConfiguredEvaluator(
            base=base,
            domain=config.domain,
            metric=metric,
            maximize=maximize,
            target=cell.target,
        )

    return load


def _operator_factory(evaluator_dir: Path) -> Callable[[str], _OperatorAdapter]:
    def build(name: str) -> _OperatorAdapter:
        from autoevolve.mutate.registry import get_operator

        return _OperatorAdapter(get_operator(name), evaluator_dir)

    return build


@contextmanager
def _cell_environment(values: Mapping[str, str]) -> Iterator[None]:
    from autoevolve.eval import contract as contract_module
    from autoevolve.eval import sandbox as sandbox_module

    prior_values = {name: os.environ.get(name) for name in values}
    prior_contract = contract_module._ALLOWED_ENV
    prior_sandbox = sandbox_module._ALLOWED_ENV
    allowed = frozenset(values)
    contract_module._ALLOWED_ENV = prior_contract | allowed
    sandbox_module._ALLOWED_ENV = prior_sandbox | allowed
    try:
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        contract_module._ALLOWED_ENV = prior_contract
        sandbox_module._ALLOWED_ENV = prior_sandbox
        for name, previous in prior_values.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _write_run_artifacts(home: Path, run_id: str) -> None:
    if not (home / "autoevolve.db").is_file():
        return
    from autoevolve.cli.render import render_all
    from autoevolve.cli.report import report

    output = artifact_dir(run_id)
    render_all(home, run_id, output)
    report(home, run_id, output / "report.md")


def _append_log_block(
    path: Path,
    *,
    run_id: str,
    cell: str,
    budget: Mapping[str, int | float | None],
    best_fitness: float | None,
    end_cause: str,
) -> None:
    timestamp = datetime.now(UTC).isoformat()
    budget_text = json.dumps(dict(budget), sort_keys=True, separators=(",", ":"))
    block = (
        f"\n## {timestamp}\n\n"
        f"- Run id: `{run_id}`\n"
        f"- Cell: `{cell}`\n"
        f"- Budget: `{budget_text}`\n"
        f"- Best fitness: {format_number(best_fitness)}\n"
        f"- End cause: `{end_cause}`\n"
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)


def _run_end_cause(engine: Any, run_id: str, failure: Exception | None) -> str:
    if failure is not None:
        return f"error:{type(failure).__name__}"
    status = engine.run_status(run_id)
    return str(status.get("status", "unknown"))


def _best_fitness(engine: Any, run_id: str) -> float | None:
    rows = engine.best(run_id, k=1)
    if not rows:
        return None
    value = rows[0].get("fitness")
    return float(value) if isinstance(value, int | float) else None


def _campaign_snapshots(home: Path, name: str) -> tuple[Snapshot, ...]:
    database = home / "autoevolve.db"
    if not database.is_file():
        return ()
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT id FROM runs WHERE goal_text LIKE ? ORDER BY created_at, id",
            (f"campaign:{name}:%",),
        ).fetchall()
    finally:
        connection.close()
    return tuple(load_snapshot(home, str(row[0])) for row in rows)


def _run_fact(snapshot: Snapshot) -> _RunFact:
    best = snapshot.best_program()
    best_value = snapshot.score(best.id) if best is not None else None
    best_scores = snapshot.scores.get(best.id, {}) if best is not None else {}
    baseline_raw = snapshot.contract.get("baseline")
    baseline = float(baseline_raw) if isinstance(baseline_raw, int | float) else None
    improved = False
    if best_value is not None and baseline is not None:
        improved = best_value > baseline if snapshot.maximize else best_value < baseline
    fitness = None
    if best_value is not None:
        fitness = best_value if snapshot.maximize else -best_value
    r2_raw = best_scores.get("r2_heldout")
    return _RunFact(
        run_id=snapshot.run.id,
        seed=snapshot.run.seed,
        completed=terminal_reason(snapshot) != "open",
        metric=snapshot.metric,
        best_value=best_value,
        best_fitness=fitness,
        improved=improved,
        r2_heldout=float(r2_raw) if isinstance(r2_raw, int | float) else None,
    )


def _best_fact(facts: list[_RunFact]) -> _RunFact | None:
    measured = [fact for fact in facts if fact.best_fitness is not None]
    if not measured:
        return facts[0] if facts else None
    return max(measured, key=lambda fact: (float(fact.best_fitness), fact.run_id))


def _labels(
    config: CampaignConfig,
    facts: list[_RunFact],
    best: _RunFact | None,
) -> tuple[str, str]:
    completed = [fact for fact in facts if fact.completed]
    improving_seeds = {fact.seed for fact in completed if fact.improved}
    replicated = len(improving_seeds) >= config.replicate_seeds
    if not completed:
        ladder = "not run"
    elif replicated:
        ladder = f"replicate-{config.replicate_seeds}"
    else:
        ladder = "proxy candidate"

    rediscovery = (
        config.name == "equation-discovery"
        and best is not None
        and best.r2_heldout is not None
        and best.r2_heldout > 0.99
    )
    if rediscovery:
        classification = "rediscovery"
    elif replicated:
        classification = "discovery"
    else:
        classification = "candidate"
    return ladder, classification


def _claims_report(violations: list[ClaimViolation], root: Path | None) -> str:
    if not violations:
        return "Claims lint: pass.\n"
    locations = ", ".join(
        f"{violation.path.relative_to(root) if root else violation.path}:"
        f"{violation.line_number}"
        for violation in violations
    )
    return f"Claims lint: blocked by {len(violations)} line(s): {locations}.\n"


def _parse_cells(value: object, path: Path) -> tuple[CampaignCell, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{path} cells must be a non-empty list")
    cells: list[CampaignCell] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise CampaignError(f"{path} cells[{index}] must be an object")
        required = {"key", "env", "target"}
        missing = sorted(required - set(raw))
        if missing:
            raise CampaignError(
                f"{path} cells[{index}] is missing keys: {', '.join(missing)}"
            )
        unknown = sorted(set(raw) - required)
        if unknown:
            raise CampaignError(
                f"{path} cells[{index}] has unknown keys: {', '.join(unknown)}"
            )
        key = _nonempty_string(raw.get("key"), f"cells[{index}].key", path)
        if ":" in key:
            raise CampaignError(f"{path} cells[{index}].key may not contain ':'")
        env_raw = raw.get("env")
        if not isinstance(env_raw, dict):
            raise CampaignError(f"{path} cells[{index}].env must be an object")
        env: dict[str, str] = {}
        for env_key, env_value in env_raw.items():
            if not isinstance(env_key, str) or not env_key:
                raise CampaignError(
                    f"{path} cells[{index}].env keys must be non-empty strings"
                )
            if not isinstance(env_value, str):
                raise CampaignError(
                    f"{path} cells[{index}].env values must be strings"
                )
            env[env_key] = env_value
        target_raw = raw.get("target")
        if target_raw is None:
            target = None
        elif isinstance(target_raw, bool) or not isinstance(target_raw, int | float):
            raise CampaignError(f"{path} cells[{index}].target must be numeric or null")
        else:
            target = float(target_raw)
        cells.append(CampaignCell(key=key, env=env, target=target))
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        raise CampaignError(f"{path} cell keys must be unique")
    return tuple(cells)


def _parse_budget(
    value: object,
    field: str,
    path: Path,
) -> dict[str, int | float | None]:
    if not isinstance(value, dict):
        raise CampaignError(f"{path} {field} must be an object")
    unknown = sorted(set(value) - _BUDGET_KEYS)
    if unknown:
        raise CampaignError(f"{path} {field} has unknown keys: {', '.join(unknown)}")
    parsed: dict[str, int | float | None] = {}
    for key, raw in value.items():
        if raw is None:
            parsed[key] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
            raise CampaignError(f"{path} {field}.{key} must be a positive number or null")
        if key == "max_evals" and not isinstance(raw, int):
            raise CampaignError(f"{path} {field}.max_evals must be a positive integer")
        parsed[key] = raw
    if not any(value is not None for value in parsed.values()):
        raise CampaignError(f"{path} {field} must include at least one budget bound")
    return parsed


def _parse_ladder(value: object, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{path} ladder must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CampaignError(f"{path} ladder entries must be non-empty strings")
    return tuple(str(item).strip() for item in value)


def _nonempty_string(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{path} {field} must be a non-empty string")
    return value.strip()


def _optional_int(value: int | float | None) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: int | float | None) -> float | None:
    return float(value) if value is not None else None


def _safe_candidate_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise CampaignError(f"unsafe candidate path: {relative}")
    target = (root / path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise CampaignError(f"unsafe candidate path: {relative}") from exc
    return target


def _abort(message: str, *, code: int = 2) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)
