"""Fetch public CVRPTW instance definitions through a Modal container.

Usage:
    modal run campaigns/vrp/fetch_instances.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"
INSTANCE_REPO = "https://github.com/VROOM-Project/vroom-scripts"
INSTANCE_REPO_COMMIT = "dc81678342050b8f9c0318e9241578c6d88eecdb"
FIXTURES_RELATIVE = Path("campaigns/vrp/evaluators/vrp/fixtures")

_DATASET_PREFIXES = (
    "benchmarks/VRPTW/solomon/",
    "benchmarks/VRPTW/homberger_200/",
    "benchmarks/VRPTW/homberger_400/",
    "benchmarks/VRPTW/homberger_600/",
    "benchmarks/VRPTW/homberger_800/",
    "benchmarks/VRPTW/homberger_1000/",
)


def _head_sha() -> str:
    """Return the local commit while surviving Modal's flat container import."""

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
    .run_commands(
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout --detach {COMMIT}",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)

app = modal.App("autoevolve-vrp-fetch-instances")


@app.function(image=image, timeout=60 * 30, cpu=2.0, memory=2048)
def fetch_bundle() -> bytes:
    """Download only public instance text files and return a fixtures archive."""

    import io
    import urllib.request
    import zipfile

    archive_url = (
        f"{INSTANCE_REPO}/archive/{INSTANCE_REPO_COMMIT}.zip"
    )
    request = urllib.request.Request(archive_url, headers={"User-Agent": "autoevolve-vrp"})
    with urllib.request.urlopen(request, timeout=120) as response:
        source_bytes = response.read()

    selected: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as source:
        for info in source.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            relative = info.filename.split("/", 1)[-1]
            prefix = next(
                (candidate for candidate in _DATASET_PREFIXES if relative.startswith(candidate)),
                None,
            )
            if prefix is None:
                continue
            dataset = prefix.rstrip("/").split("/")[-1]
            filename = Path(relative).name
            selected[f"{dataset}/{filename}"] = source.read(info)

    missing = [
        prefix.rstrip("/").split("/")[-1]
        for prefix in _DATASET_PREFIXES
        if not any(name.startswith(f"{prefix.rstrip('/').split('/')[-1]}/") for name in selected)
    ]
    if missing:
        raise RuntimeError(f"instance archive is missing datasets: {', '.join(missing)}")

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(selected):
            bundle.writestr(name, selected[name])
    return output.getvalue()


def _default_destination() -> Path:
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError as exc:
        raise RuntimeError("could not resolve the local repository root") from exc
    destination = (repo_root / FIXTURES_RELATIVE).resolve()
    if not destination.is_dir():
        raise RuntimeError(f"fixtures directory does not exist: {destination}")
    return destination


@app.local_entrypoint()
def main(destination: str | None = None) -> None:
    """Fetch remotely, then write the instance-only archive into local fixtures."""

    import io
    import zipfile

    target_root = Path(destination).resolve() if destination else _default_destination()
    target_root.mkdir(parents=True, exist_ok=True)
    payload = fetch_bundle.remote()
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            target = (target_root / info.filename).resolve()
            try:
                target.relative_to(target_root)
            except ValueError as exc:
                raise RuntimeError(f"unsafe archive member: {info.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
            written.append(info.filename)
    print(f"wrote {len(written)} public instance files to {target_root}", flush=True)
