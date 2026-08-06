"""Run the isolated kidney exchange PICEF reference solver on Modal.

Usage:
    modal run campaigns/kidney-exchange/modal_exact.py
    modal run campaigns/kidney-exchange/modal_exact.py --cell pairs-5000-frontier \
        --time-limit-s 60
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"
REFERENCE = "campaigns/kidney-exchange/reference_ilp.py"


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
    .apt_install("git", "ca-certificates")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout --detach {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
        f"cd {REPO_ROOT} && uv pip install --python .venv/bin/python 'scipy>=1.13'",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)

app = modal.App("autoevolve-kidney-exchange-exact")


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=60 * 60 * 24,
)
def exact(
    cell: str = "pairs-160-frontier",
    time_limit_s: float = 600.0,
) -> dict[str, object]:
    """Run HiGHS with a bounded solve time and return its incumbent and bound."""

    import json
    import math
    import subprocess
    import time

    allowed_cells = {
        "small-validation",
        "pairs-80-frontier",
        "pairs-160-frontier",
        "pairs-5000-frontier",
    }
    if cell not in allowed_cells:
        raise ValueError(f"cell must be one of {', '.join(sorted(allowed_cells))}")
    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be a positive finite number")

    started = time.perf_counter()
    completed = subprocess.run(
        [
            f"{REPO_ROOT}/.venv/bin/python",
            f"{REPO_ROOT}/{REFERENCE}",
            "--cell",
            cell,
            "--time-limit-s",
            str(time_limit_s),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    reference_process_wall_s = time.perf_counter() - started
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-6000:]
        raise RuntimeError(
            f"reference solve failed with exit {completed.returncode}: {stderr_tail}"
        )
    report = json.loads(completed.stdout)
    image_commit = Path(f"{REPO_ROOT}/.autoevolve-image-commit").read_text(
        encoding="utf-8"
    )
    report["image_commit"] = image_commit.strip()
    report["reference_process_wall_s"] = reference_process_wall_s
    return report


@app.local_entrypoint()
def main(
    cell: str = "pairs-160-frontier",
    time_limit_s: float = 600.0,
) -> None:
    """Launch one remote reference solve and print its complete report."""

    import json

    result = exact.remote(cell=cell, time_limit_s=time_limit_s)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
