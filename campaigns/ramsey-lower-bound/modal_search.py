"""Fan independent Ramsey search seeds across Modal CPU containers.

The image is pinned to the repository HEAD commit so Modal cannot reuse a stale
clone layer. Every worker writes through the exact full-recount certificate
writer to the shared ``autoevolve-store`` volume. Sparse workers checkpoint
each new best certificate, and every worker commits the volume in ``finally``.

Examples:
    modal run campaigns/ramsey-lower-bound/modal_search.py --mode sparse
    modal run campaigns/ramsey-lower-bound/modal_search.py --mode direct --n 43
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
CAMPAIGN = Path("campaigns/ramsey-lower-bound")


def _head_sha() -> str:
    """Return the repository commit that the remote image must contain."""

    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot pin the Modal image without the repository HEAD SHA") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git returned an invalid repository HEAD SHA: {commit!r}")
    return commit


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} /root/autoevolve",
        f"cd /root/autoevolve && git checkout {COMMIT}",
        "cd /root/autoevolve && uv sync --frozen",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-ramsey-search")


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=2.0,
    memory=4096,
    timeout=60 * 60 * 24,
    max_containers=128,
)
def run_seed(job: dict[str, object]) -> dict[str, object]:
    """Run one independent seed and persist any certificate on the volume."""

    import sys

    repo = Path("/root/autoevolve")
    campaign = repo / CAMPAIGN
    sys.path.insert(0, str(campaign))

    from search_n import search as direct_search
    from search_n import write_verified_certificate
    from search_sparse42 import search as sparse_search

    mode = str(job["mode"])
    n = int(job["n"])
    seed = int(job["seed"])
    seconds = float(job["seconds"])
    store_name = str(job["store_name"])
    start_arg = str(job.get("start", ""))
    if mode not in {"direct", "sparse"}:
        raise ValueError("mode must be 'direct' or 'sparse'")
    if not store_name or Path(store_name).name != store_name or store_name in {".", ".."}:
        raise ValueError("store_name must be one safe path segment")

    output_dir = Path("/store") / store_name / "ramsey-certificates" / mode
    output = output_dir / f"n{n}-seed{seed}.json"
    writes = 0

    def persist_direct(expected_k5: int, red: list[int]) -> None:
        nonlocal writes
        write_verified_certificate(output, n, red, expected_k5=expected_k5)
        writes += 1
        store.commit()

    def persist_sparse(expected_k4: int, red: list[int]) -> None:
        nonlocal writes
        write_verified_certificate(
            output,
            n,
            red,
            expected_k5=0,
            expected_k4=expected_k4,
        )
        writes += 1
        store.commit()

    try:
        if mode == "direct":
            best_k5, best_red = direct_search(
                n,
                seed,
                seconds,
                on_certificate=persist_direct,
            )
            if best_k5 == 0 and not output.is_file():
                persist_direct(best_k5, best_red)
            return {
                "mode": mode,
                "n": n,
                "seed": seed,
                "best_k5": best_k5,
                "certificate": str(output) if best_k5 == 0 else None,
                "writes": writes,
                "commit": COMMIT,
            }

        if start_arg:
            start = Path(start_arg)
            if not start.is_absolute():
                start = repo / start
        else:
            certificate_dir = campaign / "evaluators/ramsey/certificates/k5-frontier"
            starts = sorted(certificate_dir.glob(f"n{n}-*.json"))
            if not starts:
                raise ValueError(f"no n={n} starting certificates in {certificate_dir}")
            start = starts[seed % len(starts)]

        best_k4, best_red = sparse_search(
            n,
            seed,
            seconds,
            start,
            on_certificate=persist_sparse,
        )
        if best_k4 is not None and best_red is not None:
            persist_sparse(best_k4, best_red)
        return {
            "mode": mode,
            "n": n,
            "seed": seed,
            "best_k4": best_k4,
            "start": str(start),
            "certificate": str(output) if best_red is not None else None,
            "writes": writes,
            "commit": COMMIT,
        }
    finally:
        store.commit()


@app.local_entrypoint()
def main(
    mode: str = "sparse",
    n: int = 42,
    seed_count: int = 64,
    seed_start: int = 1,
    seconds: float = 6 * 60 * 60,
    store_name: str = "ramsey-r55",
    start: str = "",
) -> None:
    """Fan a bounded set of independent seeds across remote containers."""

    import json

    if mode not in {"direct", "sparse"}:
        raise ValueError("mode must be 'direct' or 'sparse'")
    if n < 5:
        raise ValueError("n must be at least five")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    jobs = [
        {
            "mode": mode,
            "n": n,
            "seed": seed_start + offset,
            "seconds": seconds,
            "store_name": store_name,
            "start": start,
        }
        for offset in range(seed_count)
    ]
    print(
        json.dumps(
            {
                "commit": COMMIT,
                "mode": mode,
                "n": n,
                "seed_count": seed_count,
                "seconds_per_seed": seconds,
                "store_name": store_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for result in run_seed.map(jobs, order_outputs=False):
        print(json.dumps(result, sort_keys=True), flush=True)
