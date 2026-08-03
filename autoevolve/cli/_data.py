"""Read-only SQLite reconstruction shared by all CLI presentation surfaces."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ARTIFACT_FILENAMES = {
    "dashboard": "dashboard.html",
    "gif": "evolution.gif",
    "latest_png": "latest.png",
    "mp4": "evolution.mp4",
    "poster_png": "lineage_poster.png",
    "poster_svg": "lineage_poster.svg",
    "report": "report.md",
}


class RunNotFoundError(LookupError):
    """Raised when a requested run is absent from the store."""


@dataclass(frozen=True)
class RunRow:
    """One immutable row from the runs table."""

    id: str
    goal_text: str
    domain: str
    contract_json: str
    status: str
    budget_json: str
    seed: int
    evaluator_ref: str | None
    created_at: str

    @property
    def contract(self) -> dict[str, Any]:
        return _json_object(self.contract_json)

    @property
    def budget(self) -> dict[str, Any]:
        budget = _json_object(self.budget_json)
        if budget:
            return budget
        nested = self.contract.get("budget", {})
        return nested if isinstance(nested, dict) else {}


@dataclass(frozen=True)
class ProgramRow:
    """One program in deterministic insertion order."""

    id: str
    run_id: str
    parent_id: str | None
    operator: str
    code_ref: str
    island: int
    cell_key: str | None
    created_at: str

    def as_tuple(self) -> tuple[str, str, str | None, str, str, int, str | None, str]:
        return (
            self.id,
            self.run_id,
            self.parent_id,
            self.operator,
            self.code_ref,
            self.island,
            self.cell_key,
            self.created_at,
        )


@dataclass(frozen=True)
class EventRow:
    """One append-only run event."""

    run_id: str
    seq: int
    kind: str
    payload_json: str
    created_at: str

    @property
    def payload(self) -> dict[str, Any]:
        return _json_object(self.payload_json)


@dataclass(frozen=True)
class CurvePoint:
    """Best measured metric after an evaluation."""

    eval_idx: int
    program_id: str
    value: float


@dataclass(frozen=True)
class IslandSummary:
    """Per-island measurements reconstructed from program rows."""

    island: int
    evals: int
    best_program_id: str | None
    best_value: float | None


@dataclass(frozen=True)
class Snapshot:
    """All database state needed to render one run."""

    run: RunRow
    programs: tuple[ProgramRow, ...]
    edges: tuple[tuple[str, str, str], ...]
    scores: dict[str, dict[str, float]]
    stages: dict[str, dict[str, int]]
    events: tuple[EventRow, ...]
    islands: tuple[tuple[int, str | None, str | None], ...]
    operators: tuple[tuple[str, int, int, float], ...]
    discoveries: tuple[tuple[str, str, str | None, str | None, str], ...]
    gate_failed_ids: frozenset[str]

    @property
    def contract(self) -> dict[str, Any]:
        return self.run.contract

    @property
    def metric(self) -> str:
        metric = self.contract.get("metric")
        return str(metric) if metric else "fitness"

    @property
    def gate(self) -> str:
        gate = self.contract.get("gate")
        return str(gate) if gate else "gate"

    @property
    def maximize(self) -> bool:
        return bool(self.contract.get("maximize", True))

    @property
    def program_by_id(self) -> dict[str, ProgramRow]:
        return {program.id: program for program in self.programs}

    @property
    def submissions(self) -> tuple[ProgramRow, ...]:
        return tuple(program for program in self.programs if program.operator != "seed")

    @property
    def eval_count(self) -> int:
        return len(self.submissions)

    def score(self, program_id: str, metric: str | None = None) -> float | None:
        return self.scores.get(program_id, {}).get(metric or self.metric)

    def is_scored(self, program_id: str) -> bool:
        return program_id not in self.gate_failed_ids and self.score(program_id) is not None

    def best_program(self, programs: Iterable[ProgramRow] | None = None) -> ProgramRow | None:
        candidates = self.programs if programs is None else tuple(programs)
        measured = [program for program in candidates if self.is_scored(program.id)]
        if not measured:
            return None
        direction = 1.0 if self.maximize else -1.0
        return max(
            measured,
            key=lambda program: (direction * float(self.score(program.id) or 0.0), program.id),
        )

    def curve(self) -> tuple[CurvePoint, ...]:
        best: ProgramRow | None = None
        points: list[CurvePoint] = []
        seeds = [program for program in self.programs if program.operator == "seed"]
        if seeds:
            best = self.best_program(seeds)
            if best is not None:
                value = self.score(best.id)
                if value is not None:
                    points.append(CurvePoint(0, best.id, value))
        direction = 1.0 if self.maximize else -1.0
        for eval_idx, program in enumerate(self.submissions, start=1):
            value = self.score(program.id)
            if self.is_scored(program.id) and value is not None:
                best_value = self.score(best.id) if best is not None else None
                if best is None or best_value is None or direction * value > direction * best_value:
                    best = program
            if best is not None:
                best_value = self.score(best.id)
                if best_value is not None:
                    points.append(CurvePoint(eval_idx, best.id, best_value))
        return tuple(points)

    def milestones(self) -> tuple[CurvePoint, ...]:
        points = self.curve()
        milestones: list[CurvePoint] = []
        previous: float | None = None
        for point in points:
            if previous is None or point.value != previous:
                milestones.append(point)
                previous = point.value
        return tuple(milestones)

    def island_summaries(self) -> tuple[IslandSummary, ...]:
        island_ids = {item[0] for item in self.islands}
        island_ids.update(program.island for program in self.programs)
        summaries: list[IslandSummary] = []
        for island in sorted(island_ids):
            programs = [program for program in self.programs if program.island == island]
            submissions = [program for program in programs if program.operator != "seed"]
            best = self.best_program(programs)
            summaries.append(
                IslandSummary(
                    island=island,
                    evals=len(submissions),
                    best_program_id=best.id if best else None,
                    best_value=self.score(best.id) if best else None,
                )
            )
        return tuple(summaries)

    def plateau_state(self) -> tuple[int, int, bool]:
        plateau_n = int(self.contract.get("plateau_n", 150))
        eval_by_program = {
            program.id: index for index, program in enumerate(self.submissions, start=1)
        }
        archive_indexes: list[int] = []
        for event in self.events:
            if event.kind != "archive_improved":
                continue
            program_id = event_program_id(event)
            if program_id in eval_by_program:
                archive_indexes.append(eval_by_program[program_id])
                continue
            eval_idx = event.payload.get("eval_idx")
            if isinstance(eval_idx, int):
                archive_indexes.append(eval_idx)
        milestones = self.milestones()
        fallback = milestones[-1].eval_idx if milestones else 0
        last_improvement = max(archive_indexes, default=fallback)
        idle = max(0, self.eval_count - last_improvement)
        return idle, plateau_n, idle >= plateau_n

    def elapsed_seconds(self) -> float:
        """Return recorded elapsed time from run creation to the latest event."""

        latest = self.events[-1].created_at if self.events else self.run.created_at
        try:
            start_time = datetime.fromisoformat(self.run.created_at)
            latest_time = datetime.fromisoformat(latest)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, (latest_time - start_time).total_seconds())


def home_from_env() -> Path:
    """Return the configured global store path without creating it."""

    configured = os.environ.get("AUTOEVOLVE_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".autoevolve"


def artifacts_root() -> Path:
    """Return the configured output root for run artifacts."""

    configured = os.environ.get("AUTOEVOLVE_ARTIFACTS_DIR")
    return Path(configured).expanduser() if configured else Path.cwd() / "autoevolve-runs"


def artifact_dir(run_id: str) -> Path:
    """Return the default artifact directory for a run."""

    return artifacts_root() / run_id


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    """Return all canonical artifact paths under an output directory."""

    return {key: out_dir / filename for key, filename in ARTIFACT_FILENAMES.items()}


def load_snapshot(home: Path, run_id: str) -> Snapshot:
    """Load one run from SQLite using only normative schema tables."""

    database = home / "autoevolve.db"
    if not database.is_file():
        raise RunNotFoundError(f"Run {run_id!r} was not found: {database} does not exist.")
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("BEGIN")
        run_record = connection.execute(
            "SELECT id, goal_text, domain, contract_json, status, budget_json, seed, "
            "evaluator_ref, created_at FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_record is None:
            raise RunNotFoundError(f"Run {run_id!r} was not found in {database}.")
        run = RunRow(**dict(run_record))
        programs = tuple(
            ProgramRow(**dict(row))
            for row in connection.execute(
                "SELECT id, run_id, parent_id, operator, code_ref, island, cell_key, "
                "created_at FROM programs WHERE run_id = ? ORDER BY created_at, id",
                (run_id,),
            )
        )
        edges = tuple(
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT e.child_id, e.parent_id, e.kind FROM edges e "
                "JOIN programs p ON p.id = e.child_id WHERE p.run_id = ? "
                "ORDER BY p.created_at, p.id, e.parent_id, e.kind",
                (run_id,),
            )
        )
        scores: dict[str, dict[str, float]] = {}
        stages: dict[str, dict[str, int]] = {}
        for row in connection.execute(
            "SELECT s.program_id, s.metric, s.value, s.stage FROM scores s "
            "JOIN programs p ON p.id = s.program_id WHERE p.run_id = ? "
            "ORDER BY s.program_id, s.metric, s.stage",
            (run_id,),
        ):
            program_id = str(row[0])
            metric = str(row[1])
            stage = int(row[3])
            previous_stage = stages.setdefault(program_id, {}).get(metric, -1)
            if stage >= previous_stage:
                scores.setdefault(program_id, {})[metric] = float(row[2])
                stages[program_id][metric] = stage
        events = tuple(
            EventRow(**dict(row))
            for row in connection.execute(
                "SELECT run_id, seq, kind, payload_json, created_at FROM events "
                "WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
        )
        islands = tuple(
            (int(row[0]), _optional_text(row[1]), _optional_text(row[2]))
            for row in connection.execute(
                "SELECT island_id, worker_hint, last_migration_at FROM islands "
                "WHERE run_id = ? ORDER BY island_id",
                (run_id,),
            )
        )
        operators = tuple(
            (str(row[0]), int(row[1]), int(row[2]), float(row[3]))
            for row in connection.execute(
                "SELECT name, pulls, improvements, mean_gain FROM operators "
                "WHERE domain = ? ORDER BY name",
                (run.domain,),
            )
        )
        discoveries = tuple(
            (
                str(row[0]),
                str(row[1]),
                _optional_text(row[2]),
                _optional_text(row[3]),
                str(row[4]),
            )
            for row in connection.execute(
                "SELECT id, text, source_run, source_programs, created_at "
                "FROM discoveries WHERE domain = ? ORDER BY created_at, id",
                (run.domain,),
            )
        )
    finally:
        connection.close()
    gate_failed_ids = _gate_failed_ids(events, scores, str(run.contract.get("gate", "gate")))
    return Snapshot(
        run=run,
        programs=programs,
        edges=edges,
        scores=scores,
        stages=stages,
        events=events,
        islands=islands,
        operators=operators,
        discoveries=discoveries,
        gate_failed_ids=frozenset(gate_failed_ids),
    )


def event_program_id(event: EventRow) -> str | None:
    """Extract a program identifier from the documented event payload variants."""

    payload = event.payload
    for key in ("program_id", "child_id", "candidate_id"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def terminal_reason(snapshot: Snapshot) -> str:
    """Resolve the closed reason from status and terminal events."""

    aliases = {
        "budget": "budget_exhausted",
        "closed_budget": "budget_exhausted",
        "failed_feasibility": "infeasible",
        "plateau_detected": "plateau",
    }
    status = aliases.get(snapshot.run.status, snapshot.run.status)
    if status in {"target_hit", "budget_exhausted", "plateau", "infeasible", "open"}:
        return status
    for event in reversed(snapshot.events):
        if event.kind in {"target_hit", "budget_exhausted", "plateau_detected"}:
            return "plateau" if event.kind == "plateau_detected" else event.kind
        if event.kind == "run_closed":
            payload = event.payload
            value = payload.get("reason", payload.get("status"))
            if isinstance(value, str):
                candidate = aliases.get(value, value)
                if candidate in {"target_hit", "budget_exhausted", "plateau", "infeasible"}:
                    return candidate
    feasibility = snapshot.contract.get("feasibility")
    if status == "closed" and isinstance(feasibility, dict) and feasibility:
        return "infeasible"
    return status


def why_ended(snapshot: Snapshot) -> str:
    """Explain the run state in one plain paragraph using database measurements."""

    reason = terminal_reason(snapshot)
    best = snapshot.best_program()
    best_value = snapshot.score(best.id) if best else None
    metric = snapshot.metric
    if reason == "target_hit":
        target = snapshot.contract.get("target")
        return (
            f"The run ended because it reached the target of {format_number(target)} for "
            f"{metric}. The best measured value was {format_number(best_value)} from "
            f"program {best.id if best else 'none'} after {snapshot.eval_count} evaluations."
        )
    if reason == "budget_exhausted":
        bound = _budget_description(snapshot)
        return (
            f"The run ended because {bound} was exhausted. The best measured {metric} was "
            f"{format_number(best_value)} from program {best.id if best else 'none'} after "
            f"{snapshot.eval_count} evaluations."
        )
    if reason == "plateau":
        idle, plateau_n, _ = snapshot.plateau_state()
        return (
            f"The run ended on a plateau after {idle} evaluations without an archive "
            f"improvement, meeting the configured limit of {plateau_n}. The best measured "
            f"{metric} was {format_number(best_value)} from program "
            f"{best.id if best else 'none'}."
        )
    if reason == "infeasible":
        feasibility = snapshot.contract.get("feasibility")
        ceiling, method = _ceiling_details(feasibility)
        target = snapshot.contract.get("target")
        return (
            f"The run ended as infeasible because the target of {format_number(target)} for "
            f"{metric} exceeded the measured or derived ceiling of {format_number(ceiling)}. "
            f"The ceiling analysis method was {method}."
        )
    if reason == "open":
        return (
            f"The run is still open after {snapshot.eval_count} evaluations. The current best "
            f"measured {metric} is {format_number(best_value)} from program "
            f"{best.id if best else 'none'}."
        )
    return (
        f"The run ended with status {reason}. The best measured {metric} was "
        f"{format_number(best_value)} from program {best.id if best else 'none'} after "
        f"{snapshot.eval_count} evaluations."
    )


def humanize_event(event: EventRow) -> str:
    """Convert a closed-set event into compact TUI copy."""

    labels = {
        "archive_improved": "Archive improved",
        "budget_exhausted": "Evaluation budget exhausted",
        "child_submitted": "Child submitted",
        "contract_locked": "Contract locked",
        "discovery_added": "Discovery added",
        "gate_failed": "Correctness gate failed",
        "migration": "Island migration",
        "operator_update": "Operator stats updated",
        "parent_sampled": "Parent sampled",
        "plateau_detected": "Plateau detected",
        "run_closed": "Run closed",
        "run_opened": "Run opened",
        "target_hit": "Target reached",
        "worker_joined": "Worker joined",
    }
    label = labels.get(event.kind, event.kind.replace("_", " ").capitalize())
    program_id = event_program_id(event)
    return f"{label}: {program_id}" if program_id else label


def format_number(value: Any) -> str:
    """Format database numeric values compactly and consistently."""

    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return f"{float(value):.6g}"
    return str(value)


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _gate_failed_ids(
    events: tuple[EventRow, ...],
    scores: dict[str, dict[str, float]],
    gate: str,
) -> set[str]:
    failed = {
        program_id
        for event in events
        if event.kind == "gate_failed"
        if (program_id := event_program_id(event)) is not None
    }
    for program_id, metrics in scores.items():
        gate_value = metrics.get(gate)
        if gate_value is not None and gate_value <= 0:
            failed.add(program_id)
    return failed


def _budget_description(snapshot: Snapshot) -> str:
    budget = snapshot.run.budget
    labels = {
        "max_evals": f"the evaluation budget of {format_number(budget.get('max_evals'))}",
        "wall_clock_s": (
            f"the wall-clock budget of {format_number(budget.get('wall_clock_s'))} seconds"
        ),
        "max_cost_usd": f"the cost budget of ${format_number(budget.get('max_cost_usd'))}",
    }
    for event in reversed(snapshot.events):
        if event.kind != "budget_exhausted":
            continue
        key = event.payload.get("bound", event.payload.get("budget_kind"))
        if isinstance(key, str) and key in labels and budget.get(key) is not None:
            return labels[key]
    configured = [
        labels[key]
        for key, value in budget.items()
        if key in labels and value is not None
    ]
    if len(configured) == 1:
        return configured[0]
    if configured:
        return "a configured budget bound (" + ", ".join(configured) + ")"
    return "the configured budget"


def _ceiling_details(feasibility: Any) -> tuple[Any, str]:
    if not isinstance(feasibility, dict):
        return None, "not recorded"
    nested = feasibility.get("ceiling")
    if isinstance(nested, dict):
        value = nested.get("value")
        method = nested.get("method", feasibility.get("method", "not recorded"))
        return value, str(method)
    value = feasibility.get("value", feasibility.get("ceiling_value", nested))
    return value, str(feasibility.get("method", "not recorded"))
