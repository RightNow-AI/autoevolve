"""Run one cuBLAS frontier cell on a selectable Modal GPU.

The image is keyed by the local repository HEAD SHA. The shared autoevolve Volume is
mounted at /store, and every remote call commits it in a finally block.

Example:
    modal run campaigns/cublas-frontier/modal_gpu.py \
        --cell skinny-4096x8-frontier --gpu A10G --hours 6
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
EVALUATOR = "campaigns/cublas-frontier/evaluators/cublas"
DEFAULT_GOAL = (
    "Find a correct shape-specific CUDA kernel that beats the same-run cuBLAS "
    "baseline without reducing FP32 numerical precision."
)


def _head_sha() -> str:
    """Return the local commit while surviving Modal's flat container import."""

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        return "main"
    if not (repo_root / ".git").exists():
        return "main"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    sha = result.stdout.strip()
    return sha if len(sha) == 40 else "main"


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "git")
    .pip_install("uv")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex@0.146.1",
        f"git clone {REPO} /root/autoevolve",
        f"cd /root/autoevolve && git checkout --detach {COMMIT}",
        "cd /root/autoevolve && uv sync --frozen",
        "cd /root/autoevolve && uv pip install torch "
        "--index-url https://download.pytorch.org/whl/cu128",
        "cd /root/autoevolve && uv run python -c \"import torch, triton\"",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-cublas-frontier")


def _store_root(store_name: str) -> str:
    """Return one safe problem-specific root on the shared Volume."""

    invalid_name = not store_name or Path(store_name).name != store_name
    if invalid_name or store_name in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return f"/store/{store_name}"


@app.function(
    image=image,
    volumes={"/store": store},
    gpu="A10G",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def run_search(
    cell: str = "skinny-4096x8-frontier",
    goal: str = DEFAULT_GOAL,
    budget: int = 400,
    parallel: int = 1,
    hours: float = 6.0,
    seed: int = 1,
    store_name: str = "cublas-frontier",
) -> dict[str, object]:
    """Run the bounded campaign search and persist all state on the Volume."""

    import os

    if parallel != 1:
        raise ValueError("parallel must be 1 so concurrent evaluations cannot skew timing")
    root = _store_root(store_name)
    env = dict(os.environ)
    env["AUTOEVOLVE_HOME"] = f"{root}/autoevolve"
    env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"{root}/runs"
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
    try:
        store.reload()
        completed = subprocess.run(
            command,
            cwd="/root/autoevolve",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_tail = (completed.stdout or "")[-8000:]
        stderr_tail = (completed.stderr or "")[-4000:]
        print(stdout_tail, flush=True)
        if completed.returncode != 0:
            print(stderr_tail, flush=True)
        return {
            "returncode": completed.returncode,
            "commit": COMMIT,
            "cell": cell,
            "gpu_selected_by_entrypoint": True,
            "store_root": root,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        store.commit()


@app.local_entrypoint()
def main(
    cell: str = "skinny-4096x8-frontier",
    gpu: str = "A10G",
    goal: str = DEFAULT_GOAL,
    budget: int = 400,
    parallel: int = 1,
    hours: float = 6.0,
    seed: int = 1,
    store_name: str = "cublas-frontier",
) -> None:
    """Select the GPU type at invocation and call the remote search."""

    remote = run_search.with_options(gpu=gpu)
    result = remote.remote(
        cell=cell,
        goal=goal,
        budget=budget,
        parallel=parallel,
        hours=hours,
        seed=seed,
        store_name=store_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
