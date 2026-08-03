"""Public evolution engine facade shared by CLI and MCP adapters."""

from __future__ import annotations

import inspect
import json
import os
import random
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from autoevolve.core import archive, bandit, islands, sampling
from autoevolve.core.db import connection, init_db, new_id, resolve_home, transaction, utc_now
from autoevolve.core.events import append_event, load_events, next_sequence
from autoevolve.core.evolve_blocks import validate_files
from autoevolve.core.store import ContentStore
from autoevolve.core.types import (
    Budget,
    Contract,
    Descriptor,
    EvalOutcome,
    ParentBundle,
)

Cascade = Callable[[Path, Path], EvalOutcome]
EvaluatorLoader = Callable[[Path], object]


_IGNORED_DIRS = {"__pycache__", ".git", ".venv"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _candidate_files(directory: Path) -> dict[str, str]:
    """Load a candidate's text files, skipping interpreter and VCS artifacts.

    Bytecode caches appear next to any evaluator that has been imported and
    are not part of the candidate. Only utf-8 text belongs in the store.
    """

    files: dict[str, str] = {}
    if not directory.is_dir():
        raise FileNotFoundError(f"candidate directory does not exist: {directory}")
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory)
        if any(part in _IGNORED_DIRS for part in relative.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        files[relative.as_posix()] = path.read_text(encoding="utf-8")
    return files


def _descriptor(value: Descriptor | dict[str, Any]) -> Descriptor:
    if isinstance(value, Descriptor):
        return value
    return Descriptor(**value)


class _StageZeroEvaluator:
    """Delegate evaluator attributes while exposing only its first stage."""

    def __init__(self, evaluator: object):
        stages = list(evaluator.stages)
        if not stages:
            raise ValueError("evaluator must define at least one stage")
        self.stages = [stages[0]]
        self._evaluator = evaluator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)


class Engine:
    """Own durable evolution state and enforce all U1 run semantics."""

    def __init__(
        self,
        home: Path | None = None,
        cascade: Cascade | None = None,
        evaluator_loader: EvaluatorLoader | None = None,
    ):
        self.home = resolve_home(home)
        init_db(self.home)
        self.store = ContentStore(self.home)
        self._cascade = cascade
        self._evaluator_loader = evaluator_loader

    def _load_evaluator(self, evaluator_dir: Path) -> object:
        if self._evaluator_loader is not None:
            return self._evaluator_loader(evaluator_dir)
        from autoevolve.eval.contract import load_evaluator

        return load_evaluator(evaluator_dir)

    def _run_cascade(
        self,
        evaluator_dir: Path,
        candidate_dir: Path,
        evaluator: object | None = None,
    ) -> EvalOutcome:
        if self._cascade is not None:
            return self._cascade(evaluator_dir, candidate_dir)
        from autoevolve.eval.cascade import run_cascade

        parameters = tuple(inspect.signature(run_cascade).parameters)
        if parameters and "dir" in parameters[0]:
            return run_cascade(evaluator_dir, candidate_dir)
        loaded = evaluator if evaluator is not None else self._load_evaluator(evaluator_dir)
        return run_cascade(loaded, candidate_dir)

    def _resolve_evaluator(self, goal_text: str, evaluator_ref: str | Path | None) -> Path:
        if evaluator_ref is not None:
            return Path(evaluator_ref).expanduser().resolve()
        from autoevolve.mutate.models import resolve_endpoint
        from autoevolve.synth.pipeline import synthesize

        endpoint = resolve_endpoint("strong") or resolve_endpoint("cheap")
        if endpoint is None:
            raise ValueError(
                "evaluator synthesis requires a configured model endpoint "
                "(set AUTOEVOLVE_LOCAL_BASE_URL, or OPENAI_API_KEY plus "
                "AUTOEVOLVE_MODEL)"
            )
        workdir = self.home / "generated-evaluators" / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)
        return Path(synthesize(goal_text, workdir, endpoint)).expanduser().resolve()

    def _baseline_files(self, evaluator_dir: Path, evaluator: object) -> dict[str, str]:
        provided = getattr(evaluator, "baseline_files", None)
        if provided is not None:
            return dict(provided)
        return _candidate_files(evaluator_dir / "baseline")

    def _evaluate_baseline(
        self,
        evaluator_dir: Path,
        evaluator: object,
        code_ref: str,
    ) -> list[EvalOutcome]:
        outcomes: list[EvalOutcome] = []
        baseline_evaluator = _StageZeroEvaluator(evaluator)
        with tempfile.TemporaryDirectory(prefix="autoevolve-baseline-") as raw:
            candidate_dir = Path(raw)
            self.store.materialize(code_ref, candidate_dir)
            for _ in range(3):
                outcome = self._run_cascade(
                    evaluator_dir,
                    candidate_dir,
                    baseline_evaluator,
                )
                if not outcome.gate_passed:
                    reason = outcome.error or "baseline failed the correctness gate"
                    raise RuntimeError(reason)
                outcomes.append(outcome)
        return outcomes

    def _build_contract(
        self,
        goal_text: str,
        evaluator_dir: Path,
        evaluator: object,
        budget: Budget,
        outcomes: list[EvalOutcome],
        ceiling: dict[str, Any] | None,
        target_override: float | None = None,
    ) -> tuple[Contract, dict[str, float]]:
        template = getattr(evaluator, "contract", None)
        if template is not None and not isinstance(template, Contract):
            raise TypeError("evaluator.contract must be a Contract")
        spec = getattr(evaluator, "spec", {})
        if not isinstance(spec, dict):
            spec = {}
        gate = str(evaluator.gate)

        def configured(name: str, default: Any = None) -> Any:
            if template is not None:
                return getattr(template, name)
            attr = getattr(evaluator, name, None)
            if attr is not None:
                return attr
            return spec.get(name, default)

        metric = configured("metric")
        if metric is None and ceiling is not None:
            metric = ceiling.get("metric")
        if metric is None:
            candidates = sorted(set(outcomes[0].scores) - {gate})
            if not candidates:
                candidates = sorted(outcomes[0].scores)
            if not candidates:
                raise ValueError("baseline evaluator returned no metrics")
            metric = candidates[0]
        metric = str(metric)

        keys = set(outcomes[0].scores)
        if any(set(outcome.scores) != keys for outcome in outcomes[1:]):
            raise ValueError("baseline evaluator returned inconsistent metric sets")
        medians = {
            name: float(median([float(outcome.scores[name]) for outcome in outcomes]))
            for name in sorted(keys)
        }
        if gate not in medians:
            raise ValueError(f"correctness gate metric {gate!r} missing from baseline scores")
        if metric not in medians:
            raise ValueError(f"contract metric {metric!r} missing from baseline scores")

        raw_descriptors = configured("descriptors", []) or []
        contract = Contract(
            goal=goal_text,
            domain=str(configured("domain", evaluator_dir.name)),
            metric=metric,
            maximize=bool(configured("maximize", True)),
            baseline=medians[metric],
            target=(
                float(target_override)
                if target_override is not None
                else float(configured("target"))
                if configured("target") is not None
                else None
            ),
            gate=gate,
            budget=budget,
            descriptors=[_descriptor(item) for item in raw_descriptors],
            feasibility=ceiling if ceiling is not None else configured("feasibility"),
            plateau_n=int(configured("plateau_n", 150)),
        )
        if contract.plateau_n <= 0:
            raise ValueError("contract plateau_n must be positive")
        return contract, medians

    @staticmethod
    def _validate_budget(budget: Budget) -> None:
        if not budget.is_bounded():
            raise ValueError("run budget must include at least one bound")
        for name, value in asdict(budget).items():
            if value is not None and value <= 0:
                raise ValueError(f"budget {name} must be positive when provided")

    @staticmethod
    def _target_reached(contract: Contract, value: float) -> bool:
        if contract.target is None:
            return False
        if contract.maximize:
            return value >= contract.target
        return value <= contract.target

    @staticmethod
    def _infeasible(contract: Contract, ceiling: dict[str, Any] | None) -> bool:
        if contract.target is None or ceiling is None or "value" not in ceiling:
            return False
        ceiling_value = float(ceiling["value"])
        return contract.maximize and contract.target > ceiling_value

    def open_run(
        self,
        goal_text: str,
        evaluator_ref: str | Path | None = None,
        budget: Budget | None = None,
        workers: int = 4,
        seed: int | None = None,
        target: float | None = None,
    ) -> dict[str, Any]:
        """Measure a seed, lock a contract, and create a durable island population.

        target, when given, overrides any evaluator-configured target and is
        the value the run stops on. None means maximize until budget or
        plateau.
        """

        if budget is None:
            raise ValueError("run budget is required")
        self._validate_budget(budget)
        if workers <= 0:
            raise ValueError("workers must be a positive integer")
        evaluator_dir = self._resolve_evaluator(goal_text, evaluator_ref)
        evaluator = self._load_evaluator(evaluator_dir)
        baseline_files = self._baseline_files(evaluator_dir, evaluator)
        code_ref = self.store.put(baseline_files)
        outcomes = self._evaluate_baseline(evaluator_dir, evaluator, code_ref)
        ceiling_callable = evaluator.ceiling
        ceiling = ceiling_callable()
        if ceiling is not None and not isinstance(ceiling, dict):
            raise TypeError("evaluator ceiling must return a dict or None")
        contract, baseline_scores = self._build_contract(
            goal_text,
            evaluator_dir,
            evaluator,
            budget,
            outcomes,
            ceiling,
            target_override=target,
        )
        run_id = new_id("r")
        program_id = new_id("p")
        run_seed = seed if seed is not None else uuid.uuid4().int % (2**63 - 1)
        initial_fitness = archive.fitness(contract, baseline_scores[contract.metric])
        key = archive.cell_key(contract, baseline_scores)
        status = "open"
        if self._infeasible(contract, ceiling):
            status = "infeasible"
        elif self._target_reached(contract, baseline_scores[contract.metric]):
            status = "target_hit"
        now = utc_now()

        with transaction(self.home) as conn:
            conn.execute(
                "INSERT INTO runs(id, goal_text, domain, contract_json, status, budget_json, "
                "seed, evaluator_ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    goal_text,
                    contract.domain,
                    contract.to_json(),
                    status,
                    json.dumps(asdict(budget), sort_keys=True),
                    run_seed,
                    str(evaluator_dir),
                    now,
                ),
            )
            islands.create_islands(conn, run_id, workers)
            conn.execute(
                "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
                "cell_key, created_at) VALUES (?, ?, NULL, 'seed', ?, 0, ?, ?)",
                (program_id, run_id, code_ref, key, now),
            )
            conn.executemany(
                "INSERT INTO scores(program_id, metric, value, stage, measured_at) "
                "VALUES (?, ?, ?, 0, ?)",
                [
                    (program_id, metric, value, now)
                    for metric, value in sorted(baseline_scores.items())
                ],
            )
            append_event(
                conn,
                run_id,
                "run_opened",
                {"evaluator_ref": str(evaluator_dir), "seed": run_seed, "workers": workers},
            )
            append_event(
                conn,
                run_id,
                "contract_locked",
                {"contract": json.loads(contract.to_json())},
            )
            append_event(
                conn,
                run_id,
                "archive_improved",
                {
                    "best_fitness": initial_fitness,
                    "cell_key": key,
                    "eval_idx": 0,
                    "fitness": initial_fitness,
                    "program_id": program_id,
                    "replaced_program_id": None,
                },
            )
            if status == "target_hit":
                append_event(
                    conn,
                    run_id,
                    "target_hit",
                    {"program_id": program_id, "value": baseline_scores[contract.metric]},
                )
                append_event(conn, run_id, "run_closed", {"status": status})
            elif status == "infeasible":
                append_event(conn, run_id, "run_closed", {"status": status})

        result: dict[str, Any] = {"run_id": run_id, "contract": contract}
        if status != "open":
            result["status"] = status
        if status == "infeasible":
            result["feasibility"] = ceiling
        return result

    def get_contract(self, run_id: str) -> Contract:
        """Return the immutable contract for a run."""

        with connection(self.home) as conn:
            row = conn.execute(
                "SELECT contract_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return Contract.from_json(str(row["contract_json"]))

    def join_run(self, run_id: str, runtime: str) -> dict[str, int]:
        """Assign a worker to the fixed island ring in round-robin order."""

        with transaction(self.home) as conn:
            row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            assigned = islands.assign_island(conn, run_id, runtime)
            append_event(
                conn,
                run_id,
                "worker_joined",
                {"island": assigned, "runtime": runtime},
            )
        return {"island": assigned}

    def _discovery_rows(
        self,
        conn: Any,
        domain: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT id, domain, text, source_run, source_programs, created_at "
            "FROM discoveries WHERE domain = ?",
            (domain,),
        ).fetchall()
        decoded = [
            {
                "id": str(row["id"]),
                "domain": str(row["domain"]),
                "text": str(row["text"]),
                "source_run": row["source_run"],
                "source_programs": self._decode_source_programs(row["source_programs"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]
        return sampling.rank_discoveries(decoded, query, limit)

    @staticmethod
    def _decode_source_programs(value: Any) -> list[str]:
        if value is None:
            return []
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError:
            return [str(value)]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]

    def next_parent(
        self,
        run_id: str,
        island: int,
        operators: tuple[str, ...] | None = None,
    ) -> ParentBundle:
        """Choose a deterministic parent and assemble its mutation context.

        operators restricts the bandit to arms this worker can actually run.
        Without it the selector may hint an operator the worker does not
        have, the worker substitutes a different one, and the pull is
        recorded against the substitute. The hinted arm then stays unpulled
        and is hinted forever, which silently collapses every cycle onto one
        operator and starves the bandit of feedback.
        """

        with transaction(self.home) as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            if str(run["status"]) != "open":
                raise RuntimeError(f"run {run_id} is {run['status']}")
            count = islands.island_count(conn, run_id)
            if island < 0 or island >= count:
                raise ValueError(f"invalid island {island} for run {run_id}")
            elites = archive.current_elites(conn, run_id)
            has_cross_island_elite = any(elite.program.island != island for elite in elites)
            allowed = tuple(operators) if operators else bandit.OPERATORS
            unknown = sorted(set(allowed) - set(bandit.OPERATORS))
            if unknown:
                raise ValueError(f"unknown operators: {', '.join(unknown)}")
            available_operators = tuple(
                name
                for name in allowed
                if name != "crossover" or has_cross_island_elite
            )
            if not available_operators:
                available_operators = tuple(name for name in allowed if name != "crossover")
            if not available_operators:
                raise ValueError("no operator is available for this worker")
            operator_hint = bandit.select_operator(
                conn,
                str(run["domain"]),
                available_operators,
            )
            decision_seq = next_sequence(conn, run_id)
            choice: sampling.ParentChoice | None = None
            migration = False
            migration_from: int | None = None

            if count > 1 and islands.migration_due(conn, run_id, island):
                migration_from = islands.neighbor_island(conn, run_id, island)
                neighbor_pool = [
                    elite for elite in elites if elite.program.island == migration_from
                ]
                if neighbor_pool:
                    rng = sampling.seeded_rng(int(run["seed"]), "migration", decision_seq)
                    choice = sampling.choose_from_pool(neighbor_pool, rng, "migration")
                    migration = True
                    trace = dict(choice.trace)
                    trace.update(
                        {
                            "island": island,
                            "neighbor_island": migration_from,
                            "rng_event_seq": decision_seq,
                            "rng_kind": "migration",
                            "submission_count": islands.submission_count(
                                conn, run_id, island
                            ),
                        }
                    )
                    append_event(conn, run_id, "migration", trace)
                    islands.mark_migration(conn, run_id, island)

            if choice is None:
                rng = sampling.seeded_rng(int(run["seed"]), "parent_sample", decision_seq)
                choice = sampling.choose_parent(elites, island, rng)
                rng_kind = "parent_sample"
            else:
                rng_kind = "migration"

            crossover = None
            if operator_hint == "crossover":
                alternatives = [
                    elite
                    for elite in sampling.rank_elites(elites)
                    if elite.program.id != choice.elite.program.id
                    and elite.program.island != island
                ]
                if alternatives:
                    crossover = alternatives[0]
                else:
                    operator_hint = bandit.select_operator(
                        conn,
                        str(run["domain"]),
                        tuple(
                            name
                            for name in available_operators
                            if name != "crossover"
                        ),
                    )

            sample_payload = dict(choice.trace)
            sample_payload.update(
                {
                    "crossover_parent_id": (
                        crossover.program.id if crossover is not None else None
                    ),
                    "island": island,
                    "migration": migration,
                    "migration_from": migration_from,
                    "operator_hint": operator_hint,
                    "rng_event_seq": decision_seq,
                    "rng_kind": rng_kind,
                }
            )
            parent_sample_seq = append_event(
                conn,
                run_id,
                "parent_sampled",
                sample_payload,
            )
            inspiration_elites = sampling.inspirations(
                elites,
                choice.elite.program.id,
            )
            discovery_rows = self._discovery_rows(
                conn,
                str(run["domain"]),
                str(run["goal_text"]),
            )
            best_elite = max(
                elites,
                key=lambda item: item.fitness,
                default=None,
            )
            best_scores = dict(best_elite.scores) if best_elite is not None else {}
            recent_failures = self._recent_failures(conn, run_id)

        return ParentBundle(
            parent=choice.elite.program,
            parent_files=self.store.get(choice.elite.program.code_ref),
            inspirations=[
                (elite.program, elite.scores) for elite in inspiration_elites
            ],
            discoveries=[str(row["text"]) for row in discovery_rows],
            operator_hint=operator_hint,
            crossover_parent=crossover.program if crossover is not None else None,
            crossover_files=(
                self.store.get(crossover.program.code_ref) if crossover is not None else None
            ),
            parent_sample_seq=parent_sample_seq,
            parent_scores=dict(choice.elite.scores),
            best_scores=best_scores,
            inspiration_files=[
                self.store.get(elite.program.code_ref) for elite in inspiration_elites
            ],
            recent_failures=recent_failures,
        )

    def record_operator_skip(self, run_id: str, operator: str, reason: str) -> None:
        """Charge an operator that failed to produce a proposal at all.

        A skipped cycle creates no program, so submit_child never runs and the
        bandit never learns anything. Because unpulled arms are preferred
        first, an operator that always fails is selected forever and starves
        every other arm, which is indistinguishable from that operator being
        disabled. Charging the failure is what lets the bandit route around it.
        """

        with transaction(self.home) as conn:
            run = conn.execute(
                "SELECT domain FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            bandit.update_operator(conn, str(run["domain"]), operator, 1.0, 0.0)
            append_event(
                conn,
                run_id,
                "operator_update",
                {"operator": operator, "skipped": True, "reason": reason[:500]},
            )

    @staticmethod
    def _recent_failures(conn: Any, run_id: str, limit: int = 3) -> list[str]:
        """Return the most recent gate failure reasons for operator feedback.

        A mutation that cannot see why the last attempts failed repeats them.
        """

        rows = conn.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND kind = 'gate_failed' "
            "ORDER BY seq DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        reasons: list[str] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError):
                continue
            reason = payload.get("reason") or payload.get("error")
            if isinstance(reason, str) and reason:
                reasons.append(reason)
        return reasons

    @staticmethod
    def _pending_sample(
        conn: Any,
        run_id: str,
        parent_id: str,
        parent_sample_seq: int | None = None,
    ) -> dict[str, Any] | None:
        events = load_events(conn, run_id)
        consumed = {
            int(event["payload"]["parent_sample_seq"])
            for event in events
            if event["kind"] in {"child_submitted", "gate_failed"}
            and event["payload"].get("parent_sample_seq") is not None
        }
        if parent_sample_seq is not None:
            sample = next(
                (event for event in events if event["seq"] == parent_sample_seq),
                None,
            )
            if sample is None:
                raise ValueError(
                    f"parent sample seq {parent_sample_seq} not found for run {run_id}"
                )
            if sample["kind"] != "parent_sampled":
                raise ValueError(
                    f"event seq {parent_sample_seq} is {sample['kind']}, not parent_sampled"
                )
            if sample["payload"].get("chosen_parent_id") != parent_id:
                raise ValueError(
                    f"parent sample seq {parent_sample_seq} does not select parent {parent_id}"
                )
            if parent_sample_seq in consumed:
                raise ValueError(f"parent sample seq {parent_sample_seq} is already consumed")
            return {"seq": sample["seq"], **sample["payload"]}

        for event in reversed(events):
            if event["kind"] != "parent_sampled" or event["seq"] in consumed:
                continue
            if event["payload"].get("chosen_parent_id") == parent_id:
                return {"seq": event["seq"], **event["payload"]}
        return None

    @staticmethod
    def _insert_scores(
        conn: Any,
        program_id: str,
        scores: dict[str, float],
        stage: int,
    ) -> None:
        measured_at = utc_now()
        conn.executemany(
            "INSERT INTO scores(program_id, metric, value, stage, measured_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (program_id, metric, float(value), stage, measured_at)
                for metric, value in sorted(scores.items())
            ],
        )

    @staticmethod
    def _evaluation_count(conn: Any, run_id: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM programs "
            "WHERE run_id = ? AND parent_id IS NOT NULL",
            (run_id,),
        ).fetchone()
        return int(row["count"])

    def _budget_remaining(
        self,
        conn: Any,
        run: Any,
        contract: Contract,
    ) -> dict[str, float | int | None]:
        evaluations = self._evaluation_count(conn, str(run["id"]))
        evals_remaining = None
        if contract.budget.max_evals is not None:
            evals_remaining = max(contract.budget.max_evals - evaluations, 0)
        wall_remaining = None
        if contract.budget.wall_clock_s is not None:
            created = datetime.fromisoformat(str(run["created_at"]))
            elapsed = (datetime.now(UTC) - created).total_seconds()
            wall_remaining = max(contract.budget.wall_clock_s - elapsed, 0.0)
        cost_remaining = None
        if contract.budget.max_cost_usd is not None:
            row = conn.execute(
                "SELECT COALESCE(SUM(s.value), 0) AS spent FROM scores AS s "
                "JOIN programs AS p ON p.id = s.program_id "
                "WHERE p.run_id = ? AND p.parent_id IS NOT NULL "
                "AND s.metric = 'cost_usd'",
                (run["id"],),
            ).fetchone()
            cost_remaining = max(contract.budget.max_cost_usd - float(row["spent"]), 0.0)
        return {
            "max_evals": evals_remaining,
            "wall_clock_s": wall_remaining,
            "max_cost_usd": cost_remaining,
        }

    @staticmethod
    def _is_budget_exhausted(remaining: dict[str, float | int | None]) -> bool:
        return any(value is not None and value <= 0 for value in remaining.values())

    @staticmethod
    def _plateau_state(conn: Any, run_id: str, contract: Contract) -> dict[str, Any]:
        row = conn.execute(
            "SELECT MAX(seq) AS seq FROM events "
            "WHERE run_id = ? AND kind = 'archive_improved'",
            (run_id,),
        ).fetchone()
        last_improvement = int(row["seq"]) if row["seq"] is not None else -1
        # Only gate-passing submissions count. A broken operator emitting 150
        # rejects is an operator problem, handled by penalizing its bandit arm,
        # not evidence that the search has run out of ideas.
        count_row = conn.execute(
            "SELECT COUNT(*) AS count FROM events "
            "WHERE run_id = ? AND kind = 'child_submitted' AND seq > ? "
            "AND json_extract(payload_json, '$.gate_passed') = 1",
            (run_id, last_improvement),
        ).fetchone()
        non_improving = int(count_row["count"])
        return {
            "limit": contract.plateau_n,
            "non_improving": non_improving,
            "reached": non_improving >= contract.plateau_n,
        }

    def _close_if_needed(
        self,
        conn: Any,
        run: Any,
        contract: Contract,
        program_id: str,
        gate_passed: bool,
        metric_value: float,
    ) -> tuple[str, dict[str, Any], dict[str, float | int | None]]:
        plateau = self._plateau_state(conn, str(run["id"]), contract)
        remaining = self._budget_remaining(conn, run, contract)
        status = "open"
        event_kind: str | None = None
        event_payload: dict[str, Any] = {}
        if gate_passed and self._target_reached(contract, metric_value):
            status = "target_hit"
            event_kind = "target_hit"
            event_payload = {"program_id": program_id, "value": metric_value}
        elif self._is_budget_exhausted(remaining):
            status = "budget_exhausted"
            event_kind = "budget_exhausted"
            event_payload = {"program_id": program_id, "remaining": remaining}
        if status != "open":
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run["id"]))
            append_event(conn, str(run["id"]), str(event_kind), event_payload)
            append_event(conn, str(run["id"]), "run_closed", {"status": status})
        elif plateau["reached"] and not self._plateau_announced(conn, str(run["id"])):
            # A plateau no longer ends the run. It switches sampling into
            # exploration, because closing at 150 non-improving submissions
            # abandoned budget that could still find something. The run ends
            # on target or budget only.
            append_event(
                conn,
                str(run["id"]),
                "plateau_detected",
                {"program_id": program_id, **plateau},
            )
        return status, plateau, remaining

    @staticmethod
    def _plateau_announced(conn: Any, run_id: str) -> bool:
        """Report whether this plateau episode was already announced.

        A new archive improvement ends an episode, so the check is scoped to
        events after the most recent improvement.
        """

        row = conn.execute(
            "SELECT MAX(seq) AS seq FROM events "
            "WHERE run_id = ? AND kind = 'archive_improved'",
            (run_id,),
        ).fetchone()
        since = int(row["seq"]) if row["seq"] is not None else -1
        announced = conn.execute(
            "SELECT COUNT(*) AS count FROM events "
            "WHERE run_id = ? AND kind = 'plateau_detected' AND seq > ?",
            (run_id, since),
        ).fetchone()
        return int(announced["count"]) > 0

    def submit_child(
        self,
        run_id: str,
        parent_id: str,
        operator: str,
        files: dict[str, str],
        notes: str = "",
        parent_sample_seq: int | None = None,
    ) -> dict[str, Any]:
        """Validate, evaluate, record, and account for one proposed child."""

        with connection(self.home) as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            parent_row = conn.execute(
                "SELECT * FROM programs WHERE id = ? AND run_id = ?",
                (parent_id, run_id),
            ).fetchone()
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if parent_row is None:
            raise KeyError(f"unknown parent {parent_id} for run {run_id}")
        if str(run["status"]) != "open":
            raise RuntimeError(f"run {run_id} is {run['status']}")
        if operator not in bandit.OPERATORS:
            raise ValueError(f"unknown operator: {operator}")
        contract = Contract.from_json(str(run["contract_json"]))
        parent = archive.program_from_row(parent_row)
        parent_files = self.store.get(parent.code_ref)
        block_check = validate_files(parent_files, files)
        if not block_check.valid:
            with transaction(self.home) as conn:
                sample = self._pending_sample(
                    conn,
                    run_id,
                    parent_id,
                    parent_sample_seq,
                )
                append_event(
                    conn,
                    run_id,
                    "gate_failed",
                    {
                        "operator": operator,
                        "parent_id": parent_id,
                        "parent_sample_seq": sample["seq"] if sample else None,
                        "reason": "evolve_block_violation",
                        "detail": block_check.reason,
                    },
                )
            return {"rejected": True, "reason": block_check.reason}

        code_ref = self.store.put(files)
        evaluator_ref = run["evaluator_ref"]
        if evaluator_ref is None:
            raise RuntimeError(f"run {run_id} has no evaluator reference")
        evaluator_dir = Path(str(evaluator_ref))
        evaluator = self._load_evaluator(evaluator_dir)
        with tempfile.TemporaryDirectory(prefix="autoevolve-candidate-") as raw:
            candidate_dir = Path(raw)
            self.store.materialize(code_ref, candidate_dir)
            outcome = self._run_cascade(evaluator_dir, candidate_dir, evaluator)

        gate_passed = bool(outcome.gate_passed)
        error = outcome.error
        normalized_scores = {name: float(value) for name, value in outcome.scores.items()}
        if gate_passed:
            missing = [
                name
                for name in (contract.gate, contract.metric)
                if name not in normalized_scores
            ]
            if missing:
                gate_passed = False
                error = f"evaluator omitted required metrics: {', '.join(missing)}"
        if gate_passed:
            try:
                key = archive.cell_key(contract, normalized_scores)
            except ValueError as exc:
                gate_passed = False
                error = str(exc)
                key = None
        else:
            key = None
        if not gate_passed:
            normalized_scores[contract.metric] = 0.0
            normalized_scores[contract.gate] = 0.0

        program_id = new_id("p")
        now = utc_now()
        with transaction(self.home) as conn:
            current_run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if current_run is None or str(current_run["status"]) != "open":
                status = current_run["status"] if current_run is not None else "missing"
                raise RuntimeError(f"run {run_id} became {status} during evaluation")
            sample = self._pending_sample(
                conn,
                run_id,
                parent_id,
                parent_sample_seq,
            )
            child_island = int(sample["island"]) if sample is not None else parent.island
            conn.execute(
                "INSERT INTO programs(id, run_id, parent_id, operator, code_ref, island, "
                "cell_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    program_id,
                    run_id,
                    parent_id,
                    operator,
                    code_ref,
                    child_island,
                    key,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO edges(child_id, parent_id, kind) VALUES (?, ?, 'parent')",
                (program_id, parent_id),
            )
            if sample is not None and sample.get("crossover_parent_id"):
                conn.execute(
                    "INSERT OR IGNORE INTO edges(child_id, parent_id, kind) "
                    "VALUES (?, ?, 'crossover')",
                    (program_id, sample["crossover_parent_id"]),
                )
            if sample is not None and sample.get("migration"):
                conn.execute(
                    "INSERT OR IGNORE INTO edges(child_id, parent_id, kind) "
                    "VALUES (?, ?, 'migration')",
                    (program_id, parent_id),
                )
            self._insert_scores(
                conn,
                program_id,
                normalized_scores,
                max(int(outcome.stage_reached), 0),
            )
            eval_idx = self._evaluation_count(conn, run_id)
            metric_value = normalized_scores[contract.metric]
            child_fitness = (
                archive.fitness(contract, metric_value) if gate_passed else 0.0
            )
            append_event(
                conn,
                run_id,
                "child_submitted",
                {
                    "code_ref": code_ref,
                    "fitness": child_fitness,
                    "gate_passed": gate_passed,
                    "island": child_island,
                    "notes": notes,
                    "operator": operator,
                    "parent_id": parent_id,
                    "parent_sample_seq": sample["seq"] if sample else None,
                    "program_id": program_id,
                    "scores": normalized_scores,
                },
            )
            improved = False
            parent_scores = archive.scores_for_program(conn, parent_id)
            if contract.metric not in parent_scores:
                raise RuntimeError(f"parent {parent_id} has no contract metric score")
            parent_fitness = archive.fitness(contract, parent_scores[contract.metric])
            if not gate_passed:
                # Charge the operator for the failure. Skipping this update let
                # an operator that emits broken code keep the mean gain from its
                # few successes and stay selected forever.
                bandit.update_operator(
                    conn,
                    contract.domain,
                    operator,
                    parent_fitness,
                    child_fitness,
                )
                append_event(
                    conn,
                    run_id,
                    "gate_failed",
                    {
                        "parent_id": parent_id,
                        "parent_sample_seq": sample["seq"] if sample else None,
                        "program_id": program_id,
                        "reason": error or "correctness_gate_failed",
                    },
                )
            else:
                arm = bandit.update_operator(
                    conn,
                    contract.domain,
                    operator,
                    parent_fitness,
                    child_fitness,
                )
                append_event(
                    conn,
                    run_id,
                    "operator_update",
                    {
                        "improvements": arm.improvements,
                        "mean_gain": arm.mean_gain,
                        "operator": operator,
                        "pulls": arm.pulls,
                    },
                )
                improved, existing = archive.should_replace(
                    conn,
                    run_id,
                    str(key),
                    child_fitness,
                )
                if improved:
                    old_best = archive.best_elite(conn, run_id)
                    best_fitness = max(
                        child_fitness,
                        old_best.fitness if old_best is not None else child_fitness,
                    )
                    append_event(
                        conn,
                        run_id,
                        "archive_improved",
                        {
                            "best_fitness": best_fitness,
                            "cell_key": key,
                            "eval_idx": eval_idx,
                            "fitness": child_fitness,
                            "program_id": program_id,
                            "replaced_program_id": (
                                existing.program.id if existing is not None else None
                            ),
                        },
                    )

            best = archive.best_elite(conn, run_id)
            best_fitness = best.fitness if best is not None else None
            status, plateau, remaining = self._close_if_needed(
                conn,
                current_run,
                contract,
                program_id,
                gate_passed,
                metric_value,
            )

        return {
            "program_id": program_id,
            "gate_passed": gate_passed,
            "scores": normalized_scores,
            "fitness": child_fitness,
            "archive_improved": improved,
            "best_fitness": best_fitness,
            "plateau": bool(plateau["reached"]),
            "budget_remaining": remaining,
            "status": status,
        }

    def best(self, run_id: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top current archive elites by internal fitness."""

        if k < 0:
            raise ValueError("k must be non-negative")
        with connection(self.home) as conn:
            run = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            elites = sampling.rank_elites(archive.current_elites(conn, run_id))[:k]
        return [
            {
                "program_id": elite.program.id,
                "parent_id": elite.program.parent_id,
                "operator": elite.program.operator,
                "code_ref": elite.program.code_ref,
                "island": elite.program.island,
                "cell_key": elite.cell_key,
                "scores": elite.scores,
                "fitness": elite.fitness,
            }
            for elite in elites
        ]

    def lineage(self, program_id: str) -> list[dict[str, Any]]:
        """Return all parent and crossover ancestors before the requested program."""

        with connection(self.home) as conn:
            depths = {program_id: 0}
            pending = [program_id]
            while pending:
                child_id = pending.pop()
                row = conn.execute("SELECT id FROM programs WHERE id = ?", (child_id,)).fetchone()
                if row is None:
                    if child_id == program_id:
                        raise KeyError(f"unknown program: {program_id}")
                    raise RuntimeError(f"lineage references missing program: {child_id}")
                parent_rows = conn.execute(
                    "SELECT DISTINCT parent_id FROM edges WHERE child_id = ? ORDER BY parent_id",
                    (child_id,),
                ).fetchall()
                for parent_row in parent_rows:
                    parent_id = str(parent_row["parent_id"])
                    next_depth = depths[child_id] + 1
                    if next_depth > depths.get(parent_id, -1):
                        depths[parent_id] = next_depth
                        pending.append(parent_id)

            lineage: list[tuple[int, dict[str, Any]]] = []
            for item_id, depth in depths.items():
                row = conn.execute("SELECT * FROM programs WHERE id = ?", (item_id,)).fetchone()
                if row is None:
                    raise RuntimeError(f"lineage references missing program: {item_id}")
                program = archive.program_from_row(row)
                edge_rows = conn.execute(
                    "SELECT parent_id, kind FROM edges WHERE child_id = ? "
                    "ORDER BY kind, parent_id",
                    (program.id,),
                ).fetchall()
                lineage.append(
                    (
                        depth,
                        {
                            "program_id": program.id,
                            "parent_id": program.parent_id,
                            "operator": program.operator,
                            "code_ref": program.code_ref,
                            "island": program.island,
                            "cell_key": program.cell_key,
                            "scores": archive.scores_for_program(conn, program.id),
                            "edges": [
                                {
                                    "parent_id": str(edge["parent_id"]),
                                    "kind": str(edge["kind"]),
                                }
                                for edge in edge_rows
                            ],
                        },
                    )
                )
        return [
            item
            for _, item in sorted(
                lineage,
                key=lambda entry: (
                    -entry[0],
                    str(entry[1]["program_id"]),
                ),
            )
        ]

    def discoveries(self, domain: str, query: str | None = None) -> list[dict[str, Any]]:
        """Return up to five recent and keyword-relevant domain discoveries."""

        with connection(self.home) as conn:
            return self._discovery_rows(conn, domain, query or "", limit=5)

    def _artifact_paths(self, run_id: str) -> dict[str, str]:
        configured = os.environ.get("AUTOEVOLVE_ARTIFACTS_DIR")
        root = Path(configured) if configured else Path.cwd() / "autoevolve-runs"
        run_dir = root.expanduser().resolve() / run_id
        return {
            "gif": str(run_dir / "evolution.gif"),
            "poster": str(run_dir / "lineage_poster.png"),
            "dashboard": str(run_dir / "dashboard.html"),
        }

    def run_status(self, run_id: str) -> dict[str, Any]:
        """Return event-derived progress, budgets, islands, and artifact paths."""

        with connection(self.home) as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            contract = Contract.from_json(str(run["contract_json"]))
            archive_events = load_events(conn, run_id, "archive_improved")
            curve = [
                [
                    int(event["payload"]["eval_idx"]),
                    float(event["payload"]["best_fitness"]),
                ]
                for event in archive_events
            ]
            plateau = self._plateau_state(conn, run_id, contract)
            remaining = self._budget_remaining(conn, run, contract)
            elite_counts: dict[int, int] = {}
            for elite in archive.current_elites(conn, run_id):
                elite_counts[elite.program.island] = elite_counts.get(elite.program.island, 0) + 1
            island_rows = conn.execute(
                "SELECT island_id, worker_hint, last_migration_at FROM islands "
                "WHERE run_id = ? ORDER BY island_id",
                (run_id,),
            ).fetchall()
            island_status = [
                {
                    "island": int(row["island_id"]),
                    "worker_hint": row["worker_hint"],
                    "last_migration_at": row["last_migration_at"],
                    "submissions": islands.submission_count(
                        conn,
                        run_id,
                        int(row["island_id"]),
                    ),
                    "elites": elite_counts.get(int(row["island_id"]), 0),
                }
                for row in island_rows
            ]
        return {
            "status": str(run["status"]),
            "curve": curve,
            "plateau": plateau,
            "budget_remaining": remaining,
            "islands": island_status,
            "artifacts": self._artifact_paths(run_id),
        }

    def event_count(self, run_id: str) -> int:
        """Return the current event count used by deterministic decision streams."""

        with connection(self.home) as conn:
            row = conn.execute("SELECT id, seed FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            return next_sequence(conn, run_id)

    def decision_rng(
        self,
        run_id: str,
        kind: str,
        event_seq: int | None = None,
    ) -> random.Random:
        """Return a deterministic per-event random stream for loop operators."""

        with connection(self.home) as conn:
            row = conn.execute("SELECT seed FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            seq = next_sequence(conn, run_id) if event_seq is None else event_seq
            return sampling.seeded_rng(int(row["seed"]), kind, seq)
