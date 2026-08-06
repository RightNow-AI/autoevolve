"""Run a bounded ANN campaign search on one Modal CPU container.

The image is pinned to the local repository HEAD. The shared autoevolve Volume is
mounted at /store, one problem-specific store is used, and every remote call commits
the Volume in a finally block.

Example:
    modal run campaigns/ann-search/modal_search.py \
        --cell medium-r095-frontier --hours 6 --budget 400
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"
EVALUATOR = "campaigns/ann-search/evaluators/ann"
DEFAULT_GOAL = (
    "Build a CPU nearest-neighbour index that maximizes measured query throughput "
    "while passing the selected exact recall gate."
)


def _head_sha() -> str:
    """Return local HEAD while surviving Modal's flat container import."""

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        return "main"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not read repository HEAD for Modal pinning") from exc
    commit = completed.stdout.strip()
    valid = len(commit) == 40 and all(character in "0123456789abcdef" for character in commit)
    if not valid:
        raise RuntimeError("git rev-parse returned an invalid repository HEAD")
    return commit


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex@0.146.1",
    )
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout --detach {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-ann-search")


def _store_root(store_name: str) -> str:
    """Return one safe problem-specific root on the shared Volume."""

    invalid = not store_name or Path(store_name).name != store_name
    if invalid or store_name in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return f"/store/{store_name}"


def _configure_codex(env: dict[str, str]) -> dict[str, str]:
    """Configure the verified non-interactive Codex runtime inside Modal."""

    api_key = env.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Modal secret autoevolve-model did not supply OPENAI_API_KEY")
    configured = dict(env)
    configured["CODEX_API_KEY"] = api_key
    lines = [
        "# Generated inside the Modal container. No credential is stored here.",
        'approval_policy = "never"',
    ]
    base_url = env.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        lines.append(f"openai_base_url = {json.dumps(base_url)}")
    codex_home = Path.home() / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return configured


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=8.0,
    memory=16384,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def run_search(
    cell: str = "medium-r095-frontier",
    goal: str = DEFAULT_GOAL,
    budget: int = 400,
    parallel: int = 1,
    hours: float = 6.0,
    seed: int = 1,
    store_name: str = "ann-search",
) -> dict[str, object]:
    """Run one measured search without concurrent CPU timing contention."""

    import os
    import threading

    if parallel != 1:
        raise ValueError("parallel must be 1 so concurrent evaluations cannot skew timing")
    root = _store_root(store_name)
    finished = threading.Event()
    keeper: threading.Thread | None = None
    try:
        store.reload()
        env = _configure_codex(dict(os.environ))
        env["AUTOEVOLVE_HOME"] = f"{root}/autoevolve"
        env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"{root}/runs"
        env["AUTOEVOLVE_CELL"] = cell
        env["AUTOEVOLVE_AGENT_RUNTIME"] = "codex"
        env["AUTOEVOLVE_AGENTIC_CODEX_NO_SANDBOX"] = "1"
        command = [
            "uv",
            "run",
            "autoevolve",
            "run",
            "--evaluator",
            EVALUATOR,
            "--goal",
            goal,
            "--budget-evals",
            str(budget),
            "--wall-clock-s",
            str(int(hours * 3600.0)),
            "--workers",
            str(parallel),
            "--parallel",
            str(parallel),
            "--operators",
            "diff,rewrite,agentic,crossover",
            "--seed",
            str(seed),
        ]

        def checkpoint() -> None:
            while not finished.wait(120.0):
                try:
                    store.commit()
                except Exception as exc:
                    print(f"checkpoint failed: {exc}", flush=True)

        keeper = threading.Thread(target=checkpoint, daemon=True)
        keeper.start()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout_tail = (completed.stdout or "")[-8000:]
        stderr_tail = (completed.stderr or "")[-4000:]
        image_commit = (Path(REPO_ROOT) / ".autoevolve-image-commit").read_text(
            encoding="utf-8"
        )
        print(stdout_tail, flush=True)
        if completed.returncode != 0:
            print(stderr_tail, flush=True)
        return {
            "returncode": completed.returncode,
            "image_commit": image_commit.strip(),
            "cell": cell,
            "store_root": root,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        finished.set()
        if keeper is not None:
            keeper.join(timeout=10.0)
        store.commit()


@app.local_entrypoint()
def main(
    cell: str = "medium-r095-frontier",
    goal: str = DEFAULT_GOAL,
    budget: int = 400,
    parallel: int = 1,
    hours: float = 6.0,
    seed: int = 1,
    store_name: str = "ann-search",
) -> None:
    """Run the selected remote search and print its persisted result."""

    result = run_search.remote(
        cell=cell,
        goal=goal,
        budget=budget,
        parallel=parallel,
        hours=hours,
        seed=seed,
        store_name=store_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
