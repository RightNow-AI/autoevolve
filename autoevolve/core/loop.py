"""In-process worker loop with injectable mutation operators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from autoevolve.core.engine import Engine

_MAX_CONSECUTIVE_SKIPS = 25
_LOG = logging.getLogger("autoevolve.loop")


def run_worker_loop(
    engine: Engine,
    run_id: str,
    get_operator: Callable[[str], object],
    max_cycles: int | None = None,
    island: int | None = None,
    on_submission: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run mutation cycles until closure or an optional deterministic cycle cap.

    The loop context intentionally contains only the locked contract, a
    per-cycle RNG, run metadata, and the engine home. The composition root
    (CLI or MCP worker) supplies operators that carry their own services.
    get_operator may substitute a different operator than the hint (for
    example under an allowlist); the submitted operator name is the one the
    returned operator reports, so bandit stats stay truthful.
    """

    if max_cycles is not None and max_cycles < 0:
        raise ValueError("max_cycles must be non-negative")
    if island is None:
        assignment = engine.join_run(run_id, "core-loop")
        island = assignment["island"]
    cycles = 0
    submissions = 0
    skips = 0
    consecutive_skips = 0
    last_result: dict[str, Any] | None = None

    while max_cycles is None or cycles < max_cycles:
        status = engine.run_status(run_id)
        if status["status"] != "open":
            break
        bundle = engine.next_parent(run_id, island)
        operator_name = bundle.operator_hint or "diff"
        operator = get_operator(operator_name)
        propose = getattr(operator, "propose", None)
        if not callable(propose):
            raise TypeError(f"operator {operator_name!r} has no callable propose method")
        event_seq = engine.event_count(run_id)
        context = SimpleNamespace(
            contract=engine.get_contract(run_id),
            rng=engine.decision_rng(run_id, f"operator:{operator_name}", event_seq),
            run_id=run_id,
            cycle=cycles,
            workdir=engine.home,
        )
        try:
            proposal = propose(bundle, context)
        except Exception as exc:
            if not getattr(exc, "skip_cycle", False):
                raise
            consecutive_skips += 1
            skips += 1
            cycles += 1
            _LOG.warning("skipped cycle %d on run %s: %s", cycles, run_id, exc)
            if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped cycles; last reason: {exc}"
                ) from exc
            continue
        consecutive_skips = 0
        files = getattr(proposal, "files", None)
        if not isinstance(files, dict):
            raise TypeError(f"operator {operator_name!r} returned no file mapping")
        notes = str(getattr(proposal, "notes", ""))
        actual_operator = str(getattr(operator, "name", operator_name))
        last_result = engine.submit_child(
            run_id,
            bundle.parent.id,
            actual_operator,
            files,
            notes,
            parent_sample_seq=bundle.parent_sample_seq,
        )
        cycles += 1
        if not last_result.get("rejected", False):
            submissions += 1
            if on_submission is not None:
                on_submission(last_result)

    final_status = engine.run_status(run_id)
    return {
        "run_id": run_id,
        "island": island,
        "cycles": cycles,
        "submissions": submissions,
        "skips": skips,
        "last_result": last_result,
        "status": final_status["status"],
        "artifacts": final_status["artifacts"],
    }
