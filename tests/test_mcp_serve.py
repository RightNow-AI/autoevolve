"""Transport wiring tests that never start a real MCP transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import autoevolve.mcp.server as server_module


class FakeServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, transport: str, **kwargs: object) -> None:
        self.calls.append((transport, kwargs))


def test_serve_stdio_builds_with_home_and_runs_stdio(monkeypatch: Any) -> None:
    fake_server = FakeServer()
    homes: list[Path | None] = []

    def fake_build_server(engine: object | None = None, home: Path | None = None) -> FakeServer:
        assert engine is None
        homes.append(home)
        return fake_server

    monkeypatch.setattr(server_module, "build_server", fake_build_server)
    home = Path("state-home")

    server_module.serve_stdio(home)

    assert homes == [home]
    assert fake_server.calls == [("stdio", {})]


def test_serve_http_passes_installed_sdk_transport_options(monkeypatch: Any) -> None:
    fake_server = FakeServer()
    homes: list[Path | None] = []

    def fake_build_server(engine: object | None = None, home: Path | None = None) -> FakeServer:
        assert engine is None
        homes.append(home)
        return fake_server

    monkeypatch.setattr(server_module, "build_server", fake_build_server)
    home = Path("state-home")

    server_module.serve_http(home=home, port=9001, host="0.0.0.0")

    assert homes == [home]
    assert fake_server.calls == [
        (
            "streamable-http",
            {"host": "0.0.0.0", "port": 9001, "streamable_http_path": "/mcp"},
        )
    ]
