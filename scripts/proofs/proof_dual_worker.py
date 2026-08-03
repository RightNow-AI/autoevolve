"""U8 proof 3: one run served simultaneously to a Claude Code and a Codex worker.

The autoevolve MCP server runs over streamable HTTP. Two real agent sessions
(claude -p headless and codex exec) connect to it, call join_run on the SAME
run, and work cycles through next_parent and submit_child. The proof asserts
from the db that both runtimes joined and both submitted programs.

    uv run python scripts/proofs/proof_dual_worker.py

Requires the claude and codex CLIs plus a model endpoint for the engine side.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PORT = 8747
URL = f"http://127.0.0.1:{PORT}/mcp"


def _resolve_exe(name: str, fallbacks: list[str]) -> str:
    """Resolve a real executable; Windows .cmd shims cannot exec without a shell."""

    import shutil

    found = shutil.which(name)
    if found and found.lower().endswith(".exe"):
        return found
    for candidate in fallbacks:
        if Path(candidate).is_file():
            return candidate
    if found:
        return found
    raise SystemExit(f"cannot resolve executable for {name}")


CLAUDE_EXE = _resolve_exe(
    "claude", [str(Path.home() / ".local" / "bin" / "claude.exe")]
)
CODEX_EXE = _resolve_exe(
    "codex",
    [
        str(
            Path(os.environ.get("APPDATA", ""))
            / "npm" / "node_modules" / "@openai" / "codex" / "node_modules"
            / "@openai" / "codex-win32-x64" / "vendor" / "x86_64-pc-windows-msvc"
            / "bin" / "codex.exe"
        )
    ],
)

WORKER_PROMPT = """You are an autoevolve worker. An autoevolve MCP server is connected.
Work the run {run_id} for exactly 2 cycles, then stop:
1. Call join_run with run_id "{run_id}" and runtime "{runtime}". Remember your island.
2. Twice: call next_parent with your island, read the parent files and the goal,
   make one small legal mutation (only change content between the EVOLVE-BLOCK-START
   and EVOLVE-BLOCK-END markers; try a faster implementation), and call submit_child
   with the full mutated file contents and operator "agentic".
3. Call run_status and print its best fitness and artifact paths.
Only mutate inside the EVOLVE-BLOCK markers. Never claim a score you did not get
from submit_child. Finish with the single line WORKER-DONE {runtime}."""


def _open_run(home: Path) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    os.environ["AUTOEVOLVE_HOME"] = str(home)
    from autoevolve.core.engine import Engine
    from autoevolve.core.types import Budget

    engine = Engine(home=home)
    opened = engine.open_run(
        goal_text="dual worker proof: speed up the image pipeline",
        evaluator_ref=str(REPO_ROOT / "evaluators" / "python-speedup"),
        budget=Budget(max_evals=12),
        workers=2,
        seed=99,
    )
    return str(opened["run_id"])


def _serve(home: Path) -> subprocess.Popen:
    env = {**os.environ, "AUTOEVOLVE_HOME": str(home)}
    log = open(REPO_ROOT / "scripts" / "proofs" / "serve.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["uv", "run", "autoevolve", "serve", "--http", "--port", str(PORT)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    import socket

    for _ in range(30):
        time.sleep(1)
        if proc.poll() is not None:
            raise SystemExit(
                f"MCP server died at startup; see scripts/proofs/serve.log"
                f" (exit {proc.returncode})"
            )
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return proc
        except OSError:
            continue
    proc.terminate()
    raise SystemExit("MCP server never opened the port; see scripts/proofs/serve.log")


def _claude_worker(run_id: str) -> subprocess.Popen:
    mcp_config = json.dumps(
        {"mcpServers": {"autoevolve": {"type": "http", "url": URL}}}
    )
    prompt = WORKER_PROMPT.format(run_id=run_id, runtime="claude-code")
    return subprocess.Popen(
        [
            CLAUDE_EXE, "-p", prompt,
            "--mcp-config", mcp_config,
            "--strict-mcp-config",
            "--allowedTools", "mcp__autoevolve__*",
            "--max-turns", "16",
            "--output-format", "text",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _codex_worker(run_id: str) -> subprocess.Popen:
    prompt = WORKER_PROMPT.format(run_id=run_id, runtime="codex")
    return subprocess.Popen(
        [
            CODEX_EXE, "exec",
            "--skip-git-repo-check",
            "-s", "read-only",
            "-c", f'mcp_servers.autoevolve.url="{URL}"',
            "-o", str(REPO_ROOT / "scripts" / "proofs" / "codex-worker-last.txt"),
            prompt,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> None:
    home = Path(os.environ.get("PROOF_HOME", str(Path.home() / ".autoevolve-proof3")))
    home.mkdir(parents=True, exist_ok=True)
    run_id = _open_run(home)
    print(f"opened {run_id}")

    server = _serve(home)
    try:
        claude_proc = _claude_worker(run_id)
        codex_proc = _codex_worker(run_id)
        claude_out, _ = claude_proc.communicate(timeout=900)
        codex_out, _ = codex_proc.communicate(timeout=900)
        print("--- claude tail ---")
        print("\n".join((claude_out or "").splitlines()[-6:]))
        print("--- codex tail ---")
        print("\n".join((codex_out or "").splitlines()[-6:]))
    finally:
        server.terminate()

    db = sqlite3.connect(home / "autoevolve.db")
    runtimes = [
        row[0]
        for row in db.execute(
            "select worker_hint from islands where run_id=?", (run_id,)
        )
        if row[0]
    ]
    programs = db.execute(
        "select count(*) from programs where run_id=? and operator != 'seed'", (run_id,)
    ).fetchone()[0]
    joined = " ".join(runtimes)
    assert "claude" in joined, f"claude worker never joined: {runtimes}"
    assert "codex" in joined, f"codex worker never joined: {runtimes}"
    assert programs >= 2, f"expected submissions from both workers, saw {programs}"
    print(f"PROOF-3 PASS run_id={run_id} runtimes={runtimes} programs={programs}")


if __name__ == "__main__":
    main()
