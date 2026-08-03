"""In-memory MCP coverage for the U4 Engine adapter."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mcp import Client

from autoevolve.mcp.server import build_server


@dataclass(frozen=True)
class FakeProgram:
    id: str
    run_id: str
    parent_id: str | None
    operator: str
    code_ref: str
    island: int
    cell_key: str | None
    created_at: str


@dataclass
class FakeParentBundle:
    parent: FakeProgram
    parent_files: dict[str, str]
    inspirations: list[tuple[FakeProgram, dict[str, float]]] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    operator_hint: str | None = None
    crossover_parent: FakeProgram | None = None
    crossover_files: dict[str, str] | None = None
    parent_sample_seq: int | None = None


def _program(
    program_id: str,
    *,
    parent_id: str | None = None,
    operator: str = "targeted_diff",
    island: int = 2,
) -> FakeProgram:
    return FakeProgram(
        id=program_id,
        run_id="run_01",
        parent_id=parent_id,
        operator=operator,
        code_ref=f"sha256:{program_id}",
        island=island,
        cell_key="size=2",
        created_at="2026-08-03T12:00:00Z",
    )


class FakeEngine:
    """Record exact calls and return deterministic Engine seam values."""

    def __init__(self) -> None:
        parent = _program("prog_18", parent_id="prog_seed")
        inspiration = _program("prog_11", parent_id="prog_seed", island=1)
        crossover = _program("prog_15", parent_id="prog_04", operator="crossover", island=3)
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.errors: dict[str, Exception] = {}
        self.responses: dict[str, object] = {
            "open_run": {
                "run_id": "run_01",
                "contract": {"metric": "p95_ms", "maximize": False},
            },
            "get_contract": {
                "goal": "Reduce p95 latency",
                "domain": "python-speedup",
                "metric": "p95_ms",
                "maximize": False,
                "baseline": 42.1,
                "target": 20.0,
                "gate": "exact parity",
                "budget": {"max_evals": 20, "wall_clock_s": 60.0, "max_cost_usd": 2.5},
                "descriptors": [],
                "feasibility": None,
                "plateau_n": 150,
            },
            "join_run": {"island": 2},
            "next_parent": FakeParentBundle(
                parent=parent,
                parent_files={"solution.py": "# EVOLVE-BLOCK-START\nold\n# EVOLVE-BLOCK-END\n"},
                inspirations=[(inspiration, {"p95_ms": 27.4})],
                discoveries=["Repeated parsing dominates valid children."],
                operator_hint="profile_guided",
                crossover_parent=crossover,
                crossover_files={"solution.py": "crossover body"},
                parent_sample_seq=41,
            ),
            "submit_child": {
                "program_id": "prog_19",
                "gate_passed": True,
                "scores": {"p95_ms": 24.8},
                "fitness": 24.8,
                "archive_improved": True,
                "best_fitness": 24.8,
                "plateau": False,
                "budget_remaining": {"max_evals": 19},
            },
            "best": [{"program_id": "prog_19", "fitness": 24.8}],
            "lineage": [
                {"program_id": "prog_seed", "parent_id": None},
                {"program_id": "prog_19", "parent_id": "prog_18"},
            ],
            "discoveries": [{"discovery_id": "disc_07", "summary": "Cache parsing."}],
            "run_status": {
                "status": "running",
                "curve": [[0, 42.1], [1, 24.8]],
                "plateau": False,
                "budget_remaining": {"max_evals": 19},
                "islands": {"active": 4},
                "artifacts": {
                    "gif": Path("artifacts/progress.gif"),
                    "poster": Path("artifacts/poster.png"),
                    "dashboard": Path("artifacts/dashboard.html"),
                },
            },
        }

    def _record(self, name: str, **arguments: object) -> object:
        self.calls.append((name, arguments))
        if name in self.errors:
            raise self.errors[name]
        return self.responses[name]

    def open_run(
        self,
        goal_text: str,
        evaluator_ref: str | None = None,
        budget: object | None = None,
        workers: int = 4,
        seed: int | None = None,
        target: float | None = None,
    ) -> object:
        assert budget is not None
        typed_budget = cast(Any, budget)
        self.calls.append(
            (
                "open_run",
                {
                    "goal_text": goal_text,
                    "evaluator_ref": evaluator_ref,
                    "budget": budget,
                    "workers": workers,
                    "seed": seed,
                },
            )
        )
        if "open_run" in self.errors:
            raise self.errors["open_run"]
        if not typed_budget.is_bounded():
            raise ValueError("At least one budget bound is required.")
        return self.responses["open_run"]

    def get_contract(self, run_id: str) -> object:
        return self._record("get_contract", run_id=run_id)

    def join_run(self, run_id: str, runtime: str) -> object:
        return self._record("join_run", run_id=run_id, runtime=runtime)

    def next_parent(self, run_id: str, island: int) -> object:
        return self._record("next_parent", run_id=run_id, island=island)

    def submit_child(
        self,
        run_id: str,
        parent_id: str,
        operator: str,
        files: dict[str, str],
        notes: str = "",
        parent_sample_seq: int | None = None,
    ) -> object:
        return self._record(
            "submit_child",
            run_id=run_id,
            parent_id=parent_id,
            operator=operator,
            files=files,
            notes=notes,
            parent_sample_seq=parent_sample_seq,
        )

    def best(self, run_id: str, k: int = 5) -> object:
        return self._record("best", run_id=run_id, k=k)

    def lineage(self, program_id: str) -> object:
        return self._record("lineage", program_id=program_id)

    def discoveries(self, domain: str, query: str | None = None) -> object:
        return self._record("discoveries", domain=domain, query=query)

    def run_status(self, run_id: str) -> object:
        return self._record("run_status", run_id=run_id)


def _result_payload(result: Any) -> object:
    if result.structured_content is not None:
        payload = result.structured_content
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload

    text_blocks = [block.text for block in result.content if block.type == "text"]
    decoded = [json.loads(text) for text in text_blocks]
    return decoded[0] if len(decoded) == 1 else decoded


def _call_tool(server: object, name: str, arguments: dict[str, object]) -> object:
    async def call() -> object:
        async with Client(server) as client:
            return _result_payload(await client.call_tool(name, arguments))

    return asyncio.run(call())


def test_list_tools_returns_exact_nine_names() -> None:
    async def list_names() -> list[str]:
        async with Client(build_server(engine=FakeEngine())) as client:
            result = await client.list_tools()
            return [tool.name for tool in result.tools]

    assert asyncio.run(list_names()) == [
        "open_run",
        "get_contract",
        "join_run",
        "next_parent",
        "submit_child",
        "best",
        "lineage",
        "discoveries",
        "run_status",
    ]


def test_tool_input_schemas_preserve_exact_arguments_and_defaults() -> None:
    tools = asyncio.run(build_server(engine=FakeEngine()).list_tools())
    schemas = {tool.name: tool.input_schema for tool in tools}

    open_schema = schemas["open_run"]
    assert set(open_schema["properties"]) == {
        "goal_text",
        "evaluator_ref",
        "max_evals",
        "wall_clock_s",
        "max_cost_usd",
        "workers",
        "seed",
        "target",
    }
    assert open_schema["required"] == ["goal_text"]
    assert open_schema["properties"]["workers"]["default"] == 4
    assert open_schema["properties"]["evaluator_ref"]["default"] is None
    assert open_schema["properties"]["max_evals"]["default"] is None
    assert open_schema["properties"]["wall_clock_s"]["default"] is None
    assert open_schema["properties"]["max_cost_usd"]["default"] is None
    assert open_schema["properties"]["seed"]["default"] is None

    submit_schema = schemas["submit_child"]
    assert submit_schema["required"] == ["run_id", "parent_id", "operator", "files"]
    assert submit_schema["properties"]["notes"]["default"] == ""
    assert submit_schema["properties"]["parent_sample_seq"]["default"] is None
    assert schemas["best"]["properties"]["k"]["default"] == 5
    assert schemas["discoveries"]["properties"]["query"]["default"] is None


def test_all_tools_round_trip_and_call_the_engine_once() -> None:
    engine = FakeEngine()
    server = build_server(engine=engine)
    child_files = {"solution.py": "# EVOLVE-BLOCK-START\nnew\n# EVOLVE-BLOCK-END\n"}

    async def exercise() -> dict[str, object]:
        async with Client(server) as client:
            calls = {
                "open_run": await client.call_tool(
                    "open_run",
                    {
                        "goal_text": "Reduce p95 latency",
                        "evaluator_ref": "evaluators/latency",
                        "max_evals": 20,
                        "wall_clock_s": 60.0,
                        "max_cost_usd": 2.5,
                        "workers": 3,
                        "seed": 17,
                    },
                ),
                "get_contract": await client.call_tool(
                    "get_contract", {"run_id": "run_01"}
                ),
                "join_run": await client.call_tool(
                    "join_run", {"run_id": "run_01", "runtime": "codex"}
                ),
                "next_parent": await client.call_tool(
                    "next_parent", {"run_id": "run_01", "island": 2}
                ),
                "submit_child": await client.call_tool(
                    "submit_child",
                    {
                        "run_id": "run_01",
                        "parent_id": "prog_18",
                        "operator": "profile_guided",
                        "files": child_files,
                    },
                ),
                "best": await client.call_tool("best", {"run_id": "run_01", "k": 2}),
                "lineage": await client.call_tool("lineage", {"program_id": "prog_19"}),
                "discoveries": await client.call_tool(
                    "discoveries", {"domain": "python-speedup"}
                ),
                "run_status": await client.call_tool("run_status", {"run_id": "run_01"}),
            }
            return {name: _result_payload(result) for name, result in calls.items()}

    payloads = asyncio.run(exercise())

    assert len(engine.calls) == 9
    open_name, open_arguments = engine.calls[0]
    assert open_name == "open_run"
    open_budget = cast(Any, open_arguments["budget"])
    assert open_arguments | {"budget": None} == {
        "goal_text": "Reduce p95 latency",
        "evaluator_ref": "evaluators/latency",
        "budget": None,
        "workers": 3,
        "seed": 17,
    }
    assert open_budget.max_evals == 20
    assert open_budget.wall_clock_s == 60.0
    assert open_budget.max_cost_usd == 2.5
    assert engine.calls[1:] == [
        ("get_contract", {"run_id": "run_01"}),
        ("join_run", {"run_id": "run_01", "runtime": "codex"}),
        ("next_parent", {"run_id": "run_01", "island": 2}),
        (
            "submit_child",
            {
                "run_id": "run_01",
                "parent_id": "prog_18",
                "operator": "profile_guided",
                "files": child_files,
                "notes": "",
                "parent_sample_seq": None,
            },
        ),
        ("best", {"run_id": "run_01", "k": 2}),
        ("lineage", {"program_id": "prog_19"}),
        ("discoveries", {"domain": "python-speedup", "query": None}),
        ("run_status", {"run_id": "run_01"}),
    ]

    assert payloads["open_run"] == engine.responses["open_run"]
    assert payloads["get_contract"] == {
        "goal": "Reduce p95 latency",
        "domain": "python-speedup",
        "metric": "p95_ms",
        "maximize": False,
        "baseline": 42.1,
        "target": 20.0,
        "gate": "exact parity",
        "budget": {"max_evals": 20, "wall_clock_s": 60.0, "max_cost_usd": 2.5},
        "descriptors": [],
        "feasibility": None,
        "plateau_n": 150,
    }
    assert payloads["join_run"] == {"island": 2}
    parent_payload = payloads["next_parent"]
    assert isinstance(parent_payload, dict)
    assert parent_payload["parent"]["id"] == "prog_18"
    assert parent_payload["inspirations"] == [
        {
            "program": {
                "id": "prog_11",
                "run_id": "run_01",
                "parent_id": "prog_seed",
                "operator": "targeted_diff",
                "code_ref": "sha256:prog_11",
                "island": 1,
                "cell_key": "size=2",
                "created_at": "2026-08-03T12:00:00Z",
            },
            "scores": {"p95_ms": 27.4},
            "files_excerpt": {},
        }
    ]
    assert parent_payload["crossover_parent"]["id"] == "prog_15"
    assert parent_payload["crossover_files"] == {"solution.py": "crossover body"}
    assert parent_payload["parent_sample_seq"] == 41
    assert payloads["submit_child"] == engine.responses["submit_child"]
    assert payloads["best"] == engine.responses["best"]
    assert payloads["lineage"] == engine.responses["lineage"]
    assert payloads["discoveries"] == engine.responses["discoveries"]
    status_payload = payloads["run_status"]
    assert isinstance(status_payload, dict)
    assert status_payload["artifacts"] == {
        "gif": str(Path("artifacts/progress.gif")),
        "poster": str(Path("artifacts/poster.png")),
        "dashboard": str(Path("artifacts/dashboard.html")),
    }


def test_engine_exception_becomes_structured_error_with_run_recovery() -> None:
    engine = FakeEngine()
    engine.errors["get_contract"] = ValueError("Unknown run id run_missing.")

    payload = _call_tool(
        build_server(engine=engine),
        "get_contract",
        {"run_id": "run_missing"},
    )

    assert isinstance(payload, dict)
    assert payload["error"] is True
    assert payload["kind"] == "ValueError"
    assert "List existing runs" in payload["message"]
    assert "open_run" in payload["message"]
    assert engine.calls == [("get_contract", {"run_id": "run_missing"})]


def test_list_tool_exception_uses_the_same_error_dictionary() -> None:
    engine = FakeEngine()
    engine.errors["best"] = ValueError("Unknown run id run_missing.")

    payload = _call_tool(
        build_server(engine=engine),
        "best",
        {"run_id": "run_missing", "k": 3},
    )

    assert isinstance(payload, dict)
    assert payload["error"] is True
    assert payload["kind"] == "ValueError"
    assert "List existing runs" in payload["message"]
    assert engine.calls == [("best", {"run_id": "run_missing", "k": 3})]


def test_open_run_all_none_budget_reaches_engine_and_returns_error() -> None:
    engine = FakeEngine()

    payload = _call_tool(
        build_server(engine=engine),
        "open_run",
        {"goal_text": "Improve the measured target"},
    )

    assert payload == {
        "error": True,
        "kind": "ValueError",
        "message": "At least one budget bound is required.",
    }
    assert len(engine.calls) == 1
    call_name, arguments = engine.calls[0]
    assert call_name == "open_run"
    budget = cast(Any, arguments.pop("budget"))
    assert arguments == {
        "goal_text": "Improve the measured target",
        "evaluator_ref": None,
        "workers": 4,
        "seed": None,
    }
    assert budget.max_evals is None
    assert budget.wall_clock_s is None
    assert budget.max_cost_usd is None


def test_parent_sample_sequence_is_forwarded_to_engine() -> None:
    engine = FakeEngine()
    child_files = {"solution.py": "# EVOLVE-BLOCK-START\nnew\n# EVOLVE-BLOCK-END\n"}

    payload = _call_tool(
        build_server(engine=engine),
        "submit_child",
        {
            "run_id": "run_01",
            "parent_id": "prog_18",
            "operator": "profile_guided",
            "files": child_files,
            "parent_sample_seq": 41,
        },
    )

    assert payload == engine.responses["submit_child"]
    assert engine.calls == [
        (
            "submit_child",
            {
                "run_id": "run_01",
                "parent_id": "prog_18",
                "operator": "profile_guided",
                "files": child_files,
                "notes": "",
                "parent_sample_seq": 41,
            },
        )
    ]
