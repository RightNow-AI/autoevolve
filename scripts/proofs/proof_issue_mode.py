"""U8 proof 1: GitHub issue mode end to end against a local fake GitHub API.

Everything except GitHub itself is real: the action entrypoint, the consent
gating, the engine, the sandboxed evaluator, the live model endpoint, the
renderer, and the terminal PR construction. The fake API records every call
so the proof can assert the exact flow. Run from the repo root:

    uv run python scripts/proofs/proof_issue_mode.py

Requires OPENAI_API_KEY plus AUTOEVOLVE_MODEL (or a local endpoint) for the
evolution phase. Exits 0 with a PASS line and the run id on success.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLS: list[tuple[str, str, dict]] = []


class FakeGitHub(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _reply(self, payload: dict | list, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _route(self, method: str) -> None:
        path = self.path.split("?")[0]
        body = self._read_body() if method in {"POST", "PATCH"} else {}
        CALLS.append((method, path, body))
        suffix = path.removeprefix("/repos/acme/demo")
        if method == "GET" and re.fullmatch(r"/issues/\d+", suffix):
            self._reply({"number": 41, "state": "open"})
        elif method == "POST" and re.fullmatch(r"/issues/\d+/comments", suffix):
            self._reply({"id": len(CALLS)}, status=201)
        elif method == "GET" and re.fullmatch(r"/issues/\d+/labels", suffix):
            self._reply([{"name": "evolve"}, {"name": "evolve:approved"}])
        elif method == "GET" and re.fullmatch(r"/collaborators/[^/]+/permission", suffix):
            self._reply({"permission": "admin"})
        elif method == "GET" and suffix == "":
            self._reply({"default_branch": "main"})
        elif method == "GET" and re.fullmatch(r"/branches/.+", suffix):
            self._reply({"name": "main", "commit": {"sha": "base0000000"}})
        elif method == "POST" and suffix == "/git/refs":
            self._reply({"ref": body.get("ref"), "object": {"sha": "base0000000"}}, status=201)
        elif method == "PATCH" and suffix.startswith("/git/refs/"):
            self._reply({"object": {"sha": "commit00001"}})
        elif method == "GET" and re.fullmatch(r"/git/commits/.+", suffix):
            self._reply({"sha": "base0000000", "tree": {"sha": "tree0000000"}})
        elif method == "POST" and suffix == "/git/blobs":
            self._reply({"sha": f"blob{len(CALLS):07d}"}, status=201)
        elif method == "POST" and suffix == "/git/trees":
            self._reply({"sha": "tree0000001"}, status=201)
        elif method == "POST" and suffix == "/git/commits":
            self._reply({"sha": "commit00001"}, status=201)
        elif method == "POST" and suffix == "/pulls":
            self._reply({"html_url": "http://fake.local/pr/1", "number": 1}, status=201)
        else:
            self._reply({"message": f"unhandled {method} {path}"}, status=404)

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def do_PATCH(self) -> None:
        self._route("PATCH")


def _run_action(event: dict, api_url: str, home: Path, workdir: Path) -> str:
    fd, raw_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    event_file = Path(raw_path)
    event_file.write_text(json.dumps(event), encoding="utf-8")
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "issues",
        "GITHUB_EVENT_PATH": str(event_file),
        "GITHUB_TOKEN": "proof-token",
        "GITHUB_REPOSITORY": "acme/demo",
        "GITHUB_API_URL": api_url,
        "GITHUB_WORKSPACE": str(workdir),
        "AUTOEVOLVE_HOME": str(home),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "autoevolve.gh.action"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    event_file.unlink(missing_ok=True)
    if completed.returncode != 0:
        print(completed.stdout[-3000:])
        print(completed.stderr[-3000:])
        raise SystemExit(f"action exited {completed.returncode}")
    return completed.stdout


def _comments() -> list[str]:
    return [
        body.get("body", "")
        for method, path, body in CALLS
        if method == "POST" and path.endswith("/comments")
    ]


def main() -> None:
    issue_body = (
        "Make the bundled image pipeline faster while keeping outputs identical.\n\n"
        "```autoevolve\nbudget_evals: 4\nworkers: 1\noperators: diff\n"
        "evaluator: evaluators/python-speedup\n```"
    )
    opened = {
        "action": "opened",
        "issue": {
            "number": 41,
            "title": "evolve: make the image pipeline faster",
            "body": issue_body,
            "labels": [{"id": 1, "name": "evolve"}],
            "user": {"login": "contributor"},
        },
        "repository": {"full_name": "acme/demo"},
        "sender": {"login": "contributor"},
    }
    labeled = {
        **opened,
        "action": "labeled",
        "label": {"id": 2, "name": "evolve:approved"},
        "sender": {"login": "maintainer"},
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api_url = f"http://127.0.0.1:{server.server_port}"

    with tempfile.TemporaryDirectory(prefix="autoevolve-proof-home-") as home_dir:
        home = Path(home_dir)

        _run_action(opened, api_url, home, REPO_ROOT)
        opened_comments = _comments()
        assert len(opened_comments) == 1, f"expected 1 proposal comment, saw {len(opened_comments)}"
        assert "CONTRACT" in opened_comments[0], "proposal comment must carry the contract block"
        db_path = home / "autoevolve.db"
        assert not db_path.exists() or not sqlite3.connect(db_path).execute(
            "select count(*) from runs"
        ).fetchone()[0], "opened handler must not execute anything"
        print("PASS opened: one proposal comment, zero execution")

        _run_action(labeled, api_url, home, REPO_ROOT)
        runs = sqlite3.connect(db_path).execute(
            "select id, status from runs"
        ).fetchall()
        assert len(runs) == 1, f"expected exactly one run, saw {runs}"
        run_id, status = runs[0]
        terminal_comments = _comments()[1:]
        assert terminal_comments, "approved flow must post comments"
        pr_calls = [c for c in CALLS if c[0] == "POST" and c[1].endswith("/pulls")]
        assert len(pr_calls) == 1, "approved flow must open exactly one terminal PR"
        pr_body = pr_calls[0][2].get("body", "")
        assert f"Run id: {run_id}" in pr_body, "PR body must end with the run id"
        blob_calls = [c for c in CALLS if c[0] == "POST" and c[1].endswith("/git/blobs")]
        assert blob_calls, "terminal PR must commit result files"
        print(f"PASS approved: run {run_id} status={status}, "
              f"{len(terminal_comments)} comments, PR with {len(blob_calls)} files")
        print(f"PROOF-1 PASS run_id={run_id}")

    server.shutdown()


if __name__ == "__main__":
    main()
