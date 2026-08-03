"""In-process worker loop with injectable mutation operators."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from autoevolve.core.engine import Engine


def run_worker_loop(
    engine: Engine,
    run_id: str,
    get_operator: Callable[[str], object],
    max_cycles: int | None = None,
) -> dict[str, Any]:
    """Run mutation cycles until closure or an optional deterministic cycle cap.

    The temporary U1 operator context intentionally contains only the locked
    contract, a per-cycle RNG, run metadata, and the engine home. U3 can duck-type
    this context while keeping core independent from ``autoevolve.mutate``.
    """

    if max_cycles is not None and max_cycles < 0:
        raise ValueError("max_cycles must be non-negative")
    assignment = engine.join_run(run_id, "core-loop")
    island = assignment["island"]
    cycles = 0
    submissions = 0
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
        proposal = propose(bundle, context)
        files = getattr(proposal, "files", None)
        if not isinstance(files, dict):
            raise TypeError(f"operator {operator_name!r} returned no file mapping")
        notes = str(getattr(proposal, "notes", ""))
        last_result = engine.submit_child(
            run_id,
            bundle.parent.id,
            operator_name,
            files,
            notes,
        )
        cycles += 1
        if not last_result.get("rejected", False):
            submissions += 1

    final_status = engine.run_status(run_id)
    return {
        "run_id": run_id,
        "island": island,
        "cycles": cycles,
        "submissions": submissions,
        "last_result": last_result,
        "status": final_status["status"],
        "artifacts": final_status["artifacts"],
    }
