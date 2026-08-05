"""Run the exact Ramsey annealer and its selftest on a Modal GPU.

The image is keyed by the repository HEAD SHA, so Modal cannot reuse a stale
clone layer after the code changes. The shared ``autoevolve-store`` volume is
mounted at ``/store`` and committed in a finally block on every remote call.

Examples:
    modal run campaigns/ramsey-lower-bound/modal_gpu.py --mode selftest
    modal run campaigns/ramsey-lower-bound/modal_gpu.py --mode search
    modal run campaigns/ramsey-lower-bound/modal_gpu.py --gpu L40S --seconds 7200
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
CAMPAIGN = "campaigns/ramsey-lower-bound"
DEFAULT_START_CERTIFICATE = (
    f"{CAMPAIGN}/evaluators/ramsey/certificates/k5-frontier/"
    "n42-18d390f53dcb4d27.json"
)


def _head_sha() -> str:
    """Read the exact local commit that the remote image must contain.

    Module level code runs twice: once locally to build the image, and again
    inside the container when Modal imports this file. There is no git
    repository in the container, so an unguarded call crash-loops every worker
    before it starts. The commit is only needed locally anyway, because the
    container already holds the code the build step checked out.
    """

    # Indexing parents raises IndexError while building the cwd argument when
    # Modal mounts this file flat at /root, which happens before any guarded
    # subprocess call and crash-loops the worker.
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        return "main"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    sha = result.stdout.strip()
    return sha if len(sha) == 40 else "main"


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} /root/autoevolve",
        f"cd /root/autoevolve && git checkout --detach {COMMIT}",
        "cd /root/autoevolve && uv sync --frozen",
        "cd /root/autoevolve && uv pip install torch "
        "--index-url https://download.pytorch.org/whl/cu128",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-ramsey-gpu")


def _stream_command(command: list[str]) -> tuple[int, str]:
    """Stream remote output while retaining a bounded diagnostic tail."""

    process = subprocess.Popen(
        command,
        cwd="/root/autoevolve",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail: deque[str] = deque(maxlen=240)
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    return process.wait(), "".join(tail)


def _store_output_dir(store_name: str, objective: str) -> str:
    """Resolve one problem-specific result directory on the shared volume."""

    if not store_name or Path(store_name).name != store_name or store_name in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return f"/store/{store_name}/ramsey-gpu/{objective}"


@app.function(
    image=image,
    volumes={"/store": store},
    gpu="A10G",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60 * 24,
)
def run_search(
    objective: str = "min_k5",
    n: int = 0,
    batch_size: int = 16_384,
    seconds: float = 3600.0,
    max_steps: int = 200_000,
    seed: int = 1,
    start_certificate: str = DEFAULT_START_CERTIFICATE,
    store_name: str = "ramsey",
    edge_chunk: int = 128,
) -> dict[str, object]:
    """Run the frontier or sparse-certificate objective on one Modal GPU."""

    if objective not in {"min_k5", "min_k4_at_zero_k5"}:
        raise ValueError("unknown objective")
    resolved_n = n or (43 if objective == "min_k5" else 42)
    output_dir = _store_output_dir(store_name, objective)
    command = [
        "uv",
        "run",
        "python",
        f"{CAMPAIGN}/gpu_anneal.py",
        "--objective",
        objective,
        "--n",
        str(resolved_n),
        "--batch-size",
        str(batch_size),
        "--seconds",
        str(seconds),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
        "--edge-chunk",
        str(edge_chunk),
    ]
    if start_certificate:
        command += ["--start", start_certificate]

    store.reload()
    try:
        returncode, output_tail = _stream_command(command)
        if returncode != 0:
            raise RuntimeError(f"GPU search exited with code {returncode}")
        return {
            "returncode": returncode,
            "commit": COMMIT,
            "objective": objective,
            "n": resolved_n,
            "output_dir": output_dir,
            "output_tail": output_tail,
        }
    finally:
        store.commit()


@app.function(
    image=image,
    volumes={"/store": store},
    gpu="A10G",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60,
)
def run_selftest(
    delta_states: int = 24,
    delta_flips: int = 32,
    easy_batch_size: int = 8192,
    easy_seconds: float = 180.0,
    easy_max_steps: int = 200_000,
) -> dict[str, object]:
    """Run exact delta and easy-search gates in a Modal GPU container."""

    command = [
        "uv",
        "run",
        "python",
        f"{CAMPAIGN}/gpu_anneal_selftest.py",
        "--delta-states",
        str(delta_states),
        "--delta-flips",
        str(delta_flips),
        "--easy-batch-size",
        str(easy_batch_size),
        "--easy-seconds",
        str(easy_seconds),
        "--easy-max-steps",
        str(easy_max_steps),
    ]
    store.reload()
    try:
        returncode, output_tail = _stream_command(command)
        if returncode != 0:
            raise RuntimeError(f"GPU selftest exited with code {returncode}")
        return {"returncode": returncode, "commit": COMMIT, "output_tail": output_tail}
    finally:
        store.commit()


@app.local_entrypoint()
def main(
    mode: str = "search",
    gpu: str = "A10G",
    objective: str = "min_k5",
    n: int = 0,
    batch_size: int = 16_384,
    seconds: float = 3600.0,
    max_steps: int = 200_000,
    seed: int = 1,
    start_certificate: str = DEFAULT_START_CERTIFICATE,
    store_name: str = "ramsey",
    edge_chunk: int = 128,
    delta_states: int = 24,
    delta_flips: int = 32,
    easy_batch_size: int = 8192,
    easy_seconds: float = 180.0,
    easy_max_steps: int = 200_000,
) -> None:
    """Select the GPU type at invocation and call one remote surface."""

    if mode == "search":
        remote = run_search.with_options(gpu=gpu)
        result = remote.remote(
            objective=objective,
            n=n,
            batch_size=batch_size,
            seconds=seconds,
            max_steps=max_steps,
            seed=seed,
            start_certificate=start_certificate,
            store_name=store_name,
            edge_chunk=edge_chunk,
        )
    elif mode == "selftest":
        remote = run_selftest.with_options(gpu=gpu)
        result = remote.remote(
            delta_states=delta_states,
            delta_flips=delta_flips,
            easy_batch_size=easy_batch_size,
            easy_seconds=easy_seconds,
            easy_max_steps=easy_max_steps,
        )
    else:
        raise ValueError("mode must be search or selftest")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
