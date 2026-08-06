"""Run generated kidney exchange evolution on Modal CPU containers.

Usage:
    modal run campaigns/kidney-exchange/modal_search.py
    modal run campaigns/kidney-exchange/modal_search.py --cell pairs-160-frontier
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"
EVALUATOR = "campaigns/kidney-exchange/evaluators/kidney"


def _head_sha() -> str:
    """Return the exact local commit while surviving Modal's flat import."""

    import subprocess

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        return "main"
    if not (repo_root / ".git").exists():
        return "main"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not read repository HEAD for Modal pinning") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git rev-parse returned an invalid repository HEAD")
    return commit


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .pip_install("uv")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout --detach {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-kidney-exchange-search")


def _safe_store_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return value


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=8.0,
    memory=16384,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def search(
    cell: str = "pairs-80-frontier",
    budget: int = 1200,
    parallel: int = 16,
    seed: int = 1,
    hours: float = 12.0,
    operators: str = "diff,rewrite,agentic,crossover",
    store_name: str = "kidney-exchange",
) -> dict[str, object]:
    """Run one bounded search and persist its database and artifacts."""

    import os
    import subprocess
    import threading

    allowed_cells = {
        "small-validation",
        "pairs-80-frontier",
        "pairs-160-frontier",
        "pairs-5000-frontier",
    }
    if cell not in allowed_cells:
        raise ValueError(f"cell must be one of {', '.join(sorted(allowed_cells))}")
    if budget < 1:
        raise ValueError("budget must be positive")
    if parallel < 1:
        raise ValueError("parallel must be positive")
    if hours <= 0.0:
        raise ValueError("hours must be positive")
    store_name = _safe_store_name(store_name)

    env = dict(os.environ)
    env["AUTOEVOLVE_HOME"] = f"/store/{store_name}/autoevolve"
    env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"/store/{store_name}/runs"
    env["AUTOEVOLVE_CELL"] = cell
    env["AUTOEVOLVE_AGENT_RUNTIME"] = "codex"

    command = [
        "uv",
        "run",
        "autoevolve",
        "run",
        "--evaluator",
        EVALUATOR,
        "--goal",
        "maximize valid kidney exchange transplants on the generated cell",
        "--budget-evals",
        str(budget),
        "--wall-clock-s",
        str(int(hours * 3600)),
        "--workers",
        str(parallel),
        "--parallel",
        str(parallel),
        "--operators",
        operators,
        "--seed",
        str(seed),
    ]

    finished = threading.Event()

    def checkpoint() -> None:
        while not finished.wait(120.0):
            try:
                store.commit()
            except Exception as exc:  # noqa: BLE001 - periodic persistence is best effort
                print(f"checkpoint failed: {exc}", flush=True)

    keeper = threading.Thread(target=checkpoint, daemon=True)
    store.reload()
    keeper.start()
    try:
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
        stdout_tail = (completed.stdout or "")[-6000:]
        stderr_tail = (completed.stderr or "")[-3000:]
        print("=== autoevolve stdout tail ===", flush=True)
        print(stdout_tail, flush=True)
        if completed.returncode != 0:
            print("=== autoevolve stderr tail ===", flush=True)
            print(stderr_tail, flush=True)
        print(f"=== exit {completed.returncode} ===", flush=True)
        image_commit = Path(f"{REPO_ROOT}/.autoevolve-image-commit").read_text(
            encoding="utf-8"
        )
        return {
            "returncode": completed.returncode,
            "cell": cell,
            "store_name": store_name,
            "image_commit": image_commit.strip(),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        finished.set()
        keeper.join(timeout=10.0)
        store.commit()


@app.local_entrypoint()
def main(
    cell: str = "pairs-80-frontier",
    budget: int = 1200,
    parallel: int = 16,
    seed: int = 1,
    hours: float = 12.0,
    operators: str = "diff,rewrite,agentic,crossover",
    store_name: str = "kidney-exchange",
) -> None:
    """Launch one remote search and print the returned summary."""

    import json

    result = search.remote(
        cell=cell,
        budget=budget,
        parallel=parallel,
        seed=seed,
        hours=hours,
        operators=operators,
        store_name=store_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
