"""Fan multicolor Ramsey search seeds across Modal CPU containers.

The image is pinned to the repository HEAD commit. Every remote worker writes
only through the exact full-recount certificate writer and commits the shared
``autoevolve-store`` volume in a finally block.

Examples:
    modal run campaigns/multicolor-ramsey/modal_search.py --mode direct
    modal run campaigns/multicolor-ramsey/modal_search.py --mode circulant
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
CAMPAIGN = Path("campaigns/multicolor-ramsey")


def _head_sha() -> str:
    """Return the repository commit that the remote image must contain."""

    import subprocess

    # Modal also imports this file flat at /root. The parent lookup itself must
    # be guarded because no repository path exists in that import context.
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
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
        raise RuntimeError("could not read the repository HEAD for Modal pinning") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git rev-parse returned an invalid repository HEAD")
    return commit


COMMIT = _head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} /root/autoevolve",
        f"cd /root/autoevolve && git checkout --detach {COMMIT}",
        "cd /root/autoevolve && uv sync --frozen",
        f"printf '%s' '{COMMIT}' > /root/autoevolve/.autoevolve-image-commit",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-multicolor-ramsey-search")


def _safe_store_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return value


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=2.0,
    memory=4096,
    timeout=60 * 60 * 24,
    max_containers=128,
)
def run_seed(job: dict[str, object]) -> dict[str, object]:
    """Run one independent exact-cost seed and persist a valid certificate."""

    import sys

    repo = Path("/root/autoevolve")
    campaign = repo / CAMPAIGN
    sys.path.insert(0, str(campaign))

    from search import search, write_verified_certificate

    mode = str(job["mode"])
    n = int(job["n"])
    seed = int(job["seed"])
    seconds = float(job["seconds"])
    store_name = _safe_store_name(str(job["store_name"]))
    if mode not in {"direct", "circulant"}:
        raise ValueError("mode must be direct or circulant")

    output_dir = Path("/store") / store_name / "multicolor-ramsey" / mode
    output = output_dir / f"n{n}-seed{seed}.json"
    image_commit = (repo / ".autoevolve-image-commit").read_text(encoding="utf-8").strip()
    writes = 0

    def persist(counts: tuple[int, int, int, int], colors: list[int]) -> None:
        nonlocal writes
        write_verified_certificate(output, n, colors, counts)
        writes += 1
        store.commit()

    store.reload()
    try:
        result = search(
            n=n,
            seed=seed,
            seconds=seconds,
            mode=mode,
            on_certificate=persist,
        )
        if sum(result.violations) == 0 and not output.is_file():
            persist(result.violations, result.colors)
        report = result.to_dict()
        report.pop("colors", None)
        report.update(
            {
                "certificate": str(output) if output.is_file() else None,
                "writes": writes,
                "image_commit": image_commit,
            }
        )
        return report
    finally:
        store.commit()


@app.local_entrypoint()
def main(
    mode: str = "direct",
    n: int = 49,
    seed_count: int = 64,
    seed_start: int = 1,
    seconds: float = 6 * 60 * 60,
    store_name: str = "multicolor-ramsey",
) -> None:
    """Fan a bounded collection of independent seeds across Modal."""

    import json

    if mode not in {"direct", "circulant"}:
        raise ValueError("mode must be direct or circulant")
    if n < 5:
        raise ValueError("n must be at least five")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    _safe_store_name(store_name)

    jobs = [
        {
            "mode": mode,
            "n": n,
            "seed": seed_start + offset,
            "seconds": seconds,
            "store_name": store_name,
        }
        for offset in range(seed_count)
    ]
    print(
        json.dumps(
            {
                "image_commit": COMMIT,
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
