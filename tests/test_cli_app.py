from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from autoevolve.cli.app import app
from tests.fixtures.viz.make_fixture import build_fixture

runner = CliRunner()


def test_init_scaffolds_worked_evaluator(tmp_path: Path) -> None:
    target = tmp_path / "demo-evaluator"

    result = runner.invoke(app, ["init", str(target)])

    assert result.exit_code == 0, result.output
    assert (target / "spec.md").is_file()
    assert (target / "evaluate.py").is_file()
    assert (target / "baseline" / "candidate.py").is_file()
    assert (target / "fixtures" / "cases.json").is_file()
    assert "Next steps" in result.output


def test_report_and_render_commands_use_configured_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    artifacts = tmp_path / "runs"
    run_id = build_fixture(home / "autoevolve.db")
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(home))
    monkeypatch.setenv("AUTOEVOLVE_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setattr("autoevolve.cli.render.shutil.which", lambda _: None)

    report_result = runner.invoke(app, ["report", run_id])
    render_result = runner.invoke(app, ["render", run_id])

    assert report_result.exit_code == 0, report_result.output
    assert render_result.exit_code == 0, render_result.output
    assert (artifacts / run_id / "report.md").is_file()
    assert (artifacts / run_id / "evolution.gif").is_file()
    assert (artifacts / run_id / "dashboard.html").is_file()


def test_run_and_join_lazily_wire_engine_and_worker_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeEngine:
        def __init__(self, home: Path):
            calls.append(("engine", home))

        def open_run(self, **kwargs: Any) -> dict[str, str]:
            calls.append(("open", kwargs))
            return {"run_id": "rfake000001"}

        def join_run(self, run_id: str, runtime: str) -> dict[str, int]:
            calls.append(("join", (run_id, runtime)))
            return {"island": 2}

        def run_status(self, run_id: str) -> dict[str, str]:
            calls.append(("status", run_id))
            return {"status": "budget_exhausted"}

    def fake_loop(engine, run_id, get_operator, max_cycles=None, island=None) -> None:
        calls.append(("loop", {"run_id": run_id, "get_operator": get_operator, "island": island}))

    engine_module = types.ModuleType("autoevolve.core.engine")
    engine_module.Engine = FakeEngine
    loop_module = types.ModuleType("autoevolve.core.loop")
    loop_module.run_worker_loop = fake_loop
    monkeypatch.setitem(sys.modules, "autoevolve.core.engine", engine_module)
    monkeypatch.setitem(sys.modules, "autoevolve.core.loop", loop_module)
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(tmp_path / "home"))

    import autoevolve.cli._data as data_module
    import autoevolve.cli.app as app_module

    monkeypatch.setattr(app_module, "_build_local_evaluator", lambda directory: None)
    monkeypatch.setattr(
        data_module,
        "load_snapshot",
        lambda home, run_id: types.SimpleNamespace(run=types.SimpleNamespace(evaluator_ref=None)),
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    run_result = runner.invoke(
        app,
        [
            "run",
            "--evaluator",
            str(evaluator),
            "--budget-evals",
            "5",
            "--operators",
            "diff,rewrite",
        ],
    )
    join_result = runner.invoke(app, ["join", "rfake000001", "--island", "1"])

    assert run_result.exit_code == 0, run_result.output
    assert join_result.exit_code == 0, join_result.output
    open_call = next(value for name, value in calls if name == "open")
    assert open_call["budget"].max_evals == 5
    loop_calls = [value for name, value in calls if name == "loop"]
    assert loop_calls[0]["island"] == 2
    assert loop_calls[1]["island"] == 1
    get_operator = loop_calls[0]["get_operator"]
    assert get_operator("diff").name == "diff"
    assert get_operator("crossover").name == "diff"


def test_serve_lazily_wires_stdio_and_http(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    def serve_stdio() -> None:
        calls.append(("stdio", None))

    def serve_http(port: int) -> None:
        calls.append(("http", port))

    server_module = types.ModuleType("autoevolve.mcp.server")
    server_module.serve_stdio = serve_stdio
    server_module.serve_http = serve_http
    monkeypatch.setitem(sys.modules, "autoevolve.mcp.server", server_module)

    stdio_result = runner.invoke(app, ["serve"])
    http_result = runner.invoke(app, ["serve", "--http", "--port", "9001"])

    assert stdio_result.exit_code == 0, stdio_result.output
    assert http_result.exit_code == 0, http_result.output
    assert calls == [("stdio", None), ("http", 9001)]


def test_missing_run_id_is_clear_and_nonzero(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(home))

    result = runner.invoke(app, ["report", "rmissing0000"])

    assert result.exit_code != 0
    assert "was not found" in result.output


def test_run_without_budget_exits_two(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    result = runner.invoke(app, ["run", "--evaluator", str(evaluator)])

    assert result.exit_code == 2
    assert "requires at least one budget bound" in result.output
