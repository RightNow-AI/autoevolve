"""MCP v2 adapter over the autoevolve Engine facade."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from mcp.server import MCPServer


def _optional_attribute(value: object, name: str, default: object) -> object:
    return getattr(value, name, default)


def _json_safe(value: object) -> Any:
    """Convert the documented Engine seam types to JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Engine returned a non-JSON-safe {type(value).__name__}")


def _dict_result(value: object) -> dict[str, Any]:
    result = _json_safe(value)
    if not isinstance(result, dict):
        raise TypeError(f"Engine returned {type(result).__name__}; expected dict")
    return result


def _list_result(value: object) -> list[dict[str, Any]]:
    result = _json_safe(value)
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise TypeError("Engine returned a value other than list[dict]")
    return cast(list[dict[str, Any]], result)


def _serialize_parent_bundle(bundle: object) -> dict[str, Any]:
    """Serialize ParentBundle without importing the parallel-owned Engine module."""
    typed_bundle = cast(Any, bundle)
    parent = _dict_result(typed_bundle.parent)
    parent_files = _dict_result(typed_bundle.parent_files)
    inspirations: list[dict[str, Any]] = []

    for raw_inspiration in typed_bundle.inspirations:
        if not isinstance(raw_inspiration, list | tuple) or len(raw_inspiration) not in {2, 3}:
            raise TypeError(
                "ParentBundle inspirations must contain program, scores, and optional files"
            )
        program = raw_inspiration[0]
        scores = _dict_result(raw_inspiration[1])
        excerpt_source = (
            raw_inspiration[2]
            if len(raw_inspiration) == 3
            else _optional_attribute(program, "files_excerpt", {})
        )
        inspirations.append(
            {
                "program": _dict_result(program),
                "scores": scores,
                "files_excerpt": _dict_result(excerpt_source),
            }
        )

    result: dict[str, Any] = {
        "parent": parent,
        "parent_files": parent_files,
        "inspirations": inspirations,
        "discoveries": _json_safe(typed_bundle.discoveries),
        "operator_hint": _json_safe(typed_bundle.operator_hint),
    }
    crossover_parent = typed_bundle.crossover_parent
    crossover_files = typed_bundle.crossover_files
    if crossover_parent is not None:
        result["crossover_parent"] = _dict_result(crossover_parent)
    if crossover_files is not None:
        result["crossover_files"] = _dict_result(crossover_files)
    return result


def _error_result(exc: Exception, *, identifier_lookup: bool = False) -> dict[str, Any]:
    message = str(exc) or type(exc).__name__
    missing_markers = ("unknown", "not found", "does not exist", "no such")
    if identifier_lookup and any(marker in message.lower() for marker in missing_markers):
        message = (
            f"{message} List existing runs with the autoevolve CLI, or call open_run "
            "to create a run."
        )
    return {"error": True, "kind": type(exc).__name__, "message": message}


def build_server(engine: object | None = None, home: Path | None = None) -> MCPServer:
    """Build the autoevolve MCP server with an optional injected Engine."""
    if engine is None:
        from autoevolve.core.engine import Engine

        engine = Engine(home)
    runtime_engine = cast(Any, engine)
    server = MCPServer(
        name="autoevolve",
        instructions=(
            "Read get_contract before joining a run. In every worker cycle, call next_parent, "
            "read its inspirations and discoveries, submit one child, then call run_status. "
            "Change marked files only inside EVOLVE-BLOCK regions. Check every result for an "
            "error key before continuing, and report measured scores and artifact paths exactly."
        ),
    )

    @server.tool()
    def open_run(
        goal_text: str,
        evaluator_ref: str | None = None,
        max_evals: int | None = None,
        wall_clock_s: float | None = None,
        max_cost_usd: float | None = None,
        workers: int = 4,
        seed: int | None = None,
        target: float | None = None,
    ) -> dict[str, Any]:
        """Open and lock an evolution run for a measured goal.

        Call this once when no existing run matches the work. The Engine creates the evaluator
        contract, measures its baseline, and refuses an unbounded budget.

        Returns: {"run_id": str, "contract": dict} plus any Engine feasibility fields, or
        {"error": true, "kind": str, "message": str}.

        Mistake to avoid: never leave max_evals, wall_clock_s, and max_cost_usd all unset.
        """
        try:
            from autoevolve.core.types import Budget

            budget = Budget(
                max_evals=max_evals,
                wall_clock_s=wall_clock_s,
                max_cost_usd=max_cost_usd,
            )
            return _dict_result(
                runtime_engine.open_run(
                    goal_text,
                    evaluator_ref=evaluator_ref,
                    budget=budget,
                    workers=workers,
                    seed=seed,
                    target=target,
                )
            )
        except Exception as exc:
            return _error_result(exc)

    @server.tool()
    def get_contract(run_id: str) -> dict[str, Any]:
        """Read the immutable scoring contract before doing worker work.

        Call this before join_run and reread it whenever the metric, gate, target, or budget is
        unclear. Treat every returned contract field as law for the life of the run.

        Returns: {"goal", "domain", "metric", "maximize", "baseline", "target", "gate",
        "budget", "descriptors", "feasibility", "plateau_n"}, or the standard error dict.

        Mistake to avoid: never infer or rewrite the metric, gate, target, or budget.
        """
        try:
            return _dict_result(runtime_engine.get_contract(run_id))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def join_run(run_id: str, runtime: str) -> dict[str, Any]:
        """Join a run and receive this worker's island assignment.

        Call this after get_contract and before the first next_parent call. Pass a useful runtime
        label such as claude-code, codex, or a local worker identifier.

        Returns: {"island": int}, or {"error": true, "kind": str, "message": str}.

        Mistake to avoid: never invent an island number or reuse another worker's assignment.
        """
        try:
            return _dict_result(runtime_engine.join_run(run_id, runtime))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def next_parent(run_id: str, island: int) -> dict[str, Any]:
        """Get the parent and evidence needed for one mutation cycle.

        Call this at the start of every worker cycle. Read the parent files, every inspiration,
        every discovery, the operator hint, and any crossover parent before editing.

        Returns: {"parent": Program fields, "parent_files": {path: content}, "inspirations":
        [{"program": Program fields, "scores": dict, "files_excerpt": {path: content}}],
        "discoveries": [str], "operator_hint": str|null, and optional "crossover_parent" and
        "crossover_files"}, or the standard error dict.

        Mistake to avoid: never mutate before reading inspirations and discoveries.
        """
        try:
            return _serialize_parent_bundle(runtime_engine.next_parent(run_id, island))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def submit_child(
        run_id: str,
        parent_id: str,
        operator: str,
        files: dict[str, str],
        notes: str = "",
    ) -> dict[str, Any]:
        """Submit full child file contents for evaluation and archive insertion.

        Call this once after producing one mutation from the current parent bundle. Use the
        operator selected or suggested for the cycle, and explain material reasoning in notes.

        Returns: {"program_id": str, "gate_passed": bool, "scores": dict, "fitness": number,
        "archive_improved": bool, "best_fitness": number, "plateau": bool,
        "budget_remaining": dict}, or the standard error dict.

        Mistake to avoid: never modify content outside EVOLVE-BLOCK markers. The Engine rejects it.
        """
        try:
            return _dict_result(
                runtime_engine.submit_child(run_id, parent_id, operator, files, notes)
            )
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def best(run_id: str, k: int = 5) -> list[dict[str, Any]] | dict[str, Any]:
        """Read the best measured programs in a run.

        Call this at checkpoints or after closure when you need the current leaders. The Engine
        owns ranking and returns its program records in best-first order.

        Returns: list[dict] containing up to k ranked program records, or the standard error dict.

        Mistake to avoid: never call a candidate best without using this measured ranking.
        """
        try:
            return _list_result(runtime_engine.best(run_id, k))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def lineage(program_id: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Read the recorded ancestry for one evaluated program.

        Call this when explaining how a strong program was reached or when choosing which prior
        operators and changes deserve closer study.

        Returns: list[dict] of recorded lineage nodes for program_id, or the standard error dict.

        Mistake to avoid: never reconstruct ancestry from memory or filenames.
        """
        try:
            return _list_result(runtime_engine.lineage(program_id))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    @server.tool()
    def discoveries(
        domain: str,
        query: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Search reusable measured discoveries for a domain.

        Call this before a mutation when the parent bundle needs broader evidence or when a query
        can focus the search on a failure, operator, or implementation technique.

        Returns: list[dict] of discovery records matching domain and query, or the standard error
        dict.

        Mistake to avoid: never treat a discovery as proof for the current child until it is
        measured by submit_child.
        """
        try:
            return _list_result(runtime_engine.discoveries(domain, query))
        except Exception as exc:
            return _error_result(exc)

    @server.tool()
    def run_status(run_id: str) -> dict[str, Any]:
        """Read closure state, progress, budget, and artifact paths for a run.

        Call this after every submit_child and at the end of every cycle summary. Stop requesting
        parents when status says the run is closed.

        Returns: {"status": str, "curve": [[eval_idx, best_fitness]], "plateau": bool,
        "budget_remaining": dict, "islands": object, "artifacts": {"gif": path|null,
        "poster": path|null, "dashboard": path|null}}, or the standard error dict.

        Mistake to avoid: never omit the artifact paths or continue after a closed status.
        """
        try:
            return _dict_result(runtime_engine.run_status(run_id))
        except Exception as exc:
            return _error_result(exc, identifier_lookup=True)

    return server


def serve_stdio(home: Path | None = None) -> None:
    """Serve autoevolve over the MCP stdio transport."""
    build_server(home=home).run("stdio")


def serve_http(
    home: Path | None = None,
    port: int = 8747,
    host: str = "127.0.0.1",
) -> None:
    """Serve autoevolve over MCP Streamable HTTP at /mcp."""
    build_server(home=home).run(
        "streamable-http",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )
