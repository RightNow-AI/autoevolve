"""Parallel Modal launcher for the order-six superpermutation ATSP search.

Launch with:

modal run campaigns/superpermutation/modal_atsp.py --seed-count 32 --deadline-minutes 60
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "autoevolve-superpermutation-atsp"
REPOSITORY_URL = "https://github.com/RightNow-AI/autoevolve.git"
REMOTE_REPOSITORY = "/opt/autoevolve"
RESULTS_MOUNT = "/results"
VOLUME_NAME = "autoevolve-superpermutation-atsp-results"
PUBLISHED_MATCH_LENGTH = 872
REMOTE_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_SEARCH_SECONDS = REMOTE_TIMEOUT_SECONDS - 5 * 60


def _git_directories(repo_root: Path) -> tuple[Path, ...]:
    marker = repo_root / ".git"
    if marker.is_dir():
        git_dir = marker.resolve()
    elif marker.is_file():
        marker_text = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if not marker_text.lower().startswith(prefix):
            raise RuntimeError(".git file does not contain a gitdir pointer")
        raw_path = marker_text[len(prefix) :].strip()
        candidate = Path(raw_path)
        git_dir = (
            candidate.resolve()
            if candidate.is_absolute()
            else (repo_root / candidate).resolve()
        )
    else:
        raise RuntimeError("repository metadata is unavailable; refusing an unpinned image")

    directories = [git_dir]
    common_marker = git_dir / "commondir"
    if common_marker.is_file():
        raw_common = common_marker.read_text(encoding="utf-8").strip()
        common_candidate = Path(raw_common)
        common_dir = (
            common_candidate.resolve()
            if common_candidate.is_absolute()
            else (git_dir / common_candidate).resolve()
        )
        if common_dir not in directories:
            directories.append(common_dir)
    return tuple(directories)


def _is_commit_sha(value: str) -> bool:
    hexadecimal = "0123456789abcdef"
    return len(value) == 40 and all(character in hexadecimal for character in value)


def _read_ref(directories: tuple[Path, ...], ref_name: str) -> str:
    for directory in directories:
        loose_ref = directory / ref_name
        if loose_ref.is_file():
            value = loose_ref.read_text(encoding="utf-8").strip().lower()
            if _is_commit_sha(value):
                return value

    for directory in directories:
        packed_refs = directory / "packed-refs"
        if not packed_refs.is_file():
            continue
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            fields = line.split(" ", maxsplit=1)
            if len(fields) != 2:
                continue
            value, packed_name = fields
            value = value.lower()
            if packed_name == ref_name and _is_commit_sha(value):
                return value
    raise RuntimeError(f"cannot resolve repository ref {ref_name!r}")


def _read_repo_head_sha(repo_root: Path) -> str:
    """Read HEAD without invoking git on the founder laptop."""

    directories = _git_directories(repo_root)
    head_value = (directories[0] / "HEAD").read_text(encoding="utf-8").strip()
    if head_value.startswith("ref: "):
        return _read_ref(directories, head_value.removeprefix("ref: ").strip())
    head_value = head_value.lower()
    if not _is_commit_sha(head_value):
        raise RuntimeError("repository HEAD is not a full commit SHA")
    return head_value


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_HEAD_SHA = _read_repo_head_sha(REPO_ROOT)

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        f"{REPOSITORY_URL} {REMOTE_REPOSITORY}",
        f"git -C {REMOTE_REPOSITORY} fetch --depth=1 origin {REPO_HEAD_SHA}",
        f"git -C {REMOTE_REPOSITORY} checkout --detach {REPO_HEAD_SHA}",
        f'test "$(git -C {REMOTE_REPOSITORY} rev-parse HEAD)" = "{REPO_HEAD_SHA}"',
    )
    .pip_install("numpy>=1.26,<3")
    .env(
        {
            "AUTOEVOLVE_MODAL_ATSP": "1",
            "AUTOEVOLVE_REPO_HEAD_SHA": REPO_HEAD_SHA,
            "PYTHONPATH": REMOTE_REPOSITORY,
        }
    )
    .workdir(REMOTE_REPOSITORY)
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")


@app.function(
    image=image,
    cpu=4.0,
    memory=4096,
    timeout=REMOTE_TIMEOUT_SECONDS,
    volumes={RESULTS_MOUNT: results_volume},
)
def run_seed(job: dict[str, int]) -> dict[str, object]:
    """Run one independent seed entirely inside a Modal CPU container."""

    seed = job["seed"]
    deadline_seconds = job["deadline_seconds"]
    if deadline_seconds <= 0 or deadline_seconds > MAX_SEARCH_SECONDS:
        raise ValueError(
            f"deadline_seconds must be between 1 and {MAX_SEARCH_SECONDS}"
        )

    if REMOTE_REPOSITORY not in sys.path:
        sys.path.insert(0, REMOTE_REPOSITORY)
    from campaigns.superpermutation.atsp_search import search, verify_superpermutation

    result = search(deadline_seconds=deadline_seconds, seed=seed)
    report = verify_superpermutation(
        result.superpermutation,
        order=result.verification.order,
        reported_length=result.verification.actual_length,
    )
    if not report.valid:
        print(
            "VERIFICATION FAILURE in Modal result handling. Refusing to report a length.",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError("exact verification failed in Modal result handling")

    artifact_path: str | None = None
    if report.actual_length < PUBLISHED_MATCH_LENGTH:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        directory_name = (
            f"{timestamp}-sha-{REPO_HEAD_SHA[:12]}-seed-{seed}-length-{report.actual_length}"
        )
        output_dir = Path(RESULTS_MOUNT) / "candidates" / directory_name
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "superpermutation.txt").write_text(
            result.superpermutation + "\n", encoding="utf-8"
        )
        _write_json(output_dir / "verification.json", report.to_dict())
        _write_json(
            output_dir / "result.json",
            {
                "repo_head_sha": REPO_HEAD_SHA,
                "seed": seed,
                "deadline_seconds": deadline_seconds,
                "elapsed_seconds": result.elapsed_seconds,
                "restarts": result.restarts,
                "length": report.actual_length,
                "verified": True,
                "path": result.path,
            },
        )
        results_volume.commit()
        artifact_path = str(output_dir)

    payload: dict[str, object] = {
        "repo_head_sha": REPO_HEAD_SHA,
        "seed": seed,
        "length": report.actual_length,
        "verified": True,
        "artifact_path": artifact_path,
        "superpermutation": result.superpermutation,
        "verification": report.to_dict(),
    }
    print(
        json.dumps(
            {
                "event": "atsp_seed_complete",
                "repo_head_sha": REPO_HEAD_SHA,
                "seed": seed,
                "length": report.actual_length,
                "verified": True,
                "artifact_path": artifact_path,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


@app.local_entrypoint()
def main(
    seed_count: int = 32,
    first_seed: int = 0,
    deadline_minutes: float = 60.0,
) -> None:
    """Launch parallel remote seeds and perform no local optimization."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    deadline_seconds = int(deadline_minutes * 60)
    if deadline_seconds <= 0 or deadline_seconds > MAX_SEARCH_SECONDS:
        raise ValueError(
            f"deadline_minutes must describe between 1 and {MAX_SEARCH_SECONDS} seconds"
        )

    jobs = [
        {"seed": first_seed + offset, "deadline_seconds": deadline_seconds}
        for offset in range(seed_count)
    ]
    local_repository = str(REPO_ROOT)
    if local_repository not in sys.path:
        sys.path.insert(0, local_repository)
    from campaigns.superpermutation.atsp_search import verify_superpermutation

    for payload in run_seed.map(jobs, order_outputs=False):
        candidate = payload.get("superpermutation")
        verification = payload.get("verification")
        if not isinstance(candidate, str) or not isinstance(verification, dict):
            print(
                "VERIFICATION FAILURE in local result receipt. Refusing to report a length.",
                file=sys.stderr,
            )
            raise RuntimeError("Modal result omitted its exact certificate")
        reported_length = verification.get("actual_length")
        if not isinstance(reported_length, int):
            print(
                "VERIFICATION FAILURE in local result receipt. Refusing to report a length.",
                file=sys.stderr,
            )
            raise RuntimeError("Modal result omitted its reported length")
        report = verify_superpermutation(candidate, reported_length=reported_length)
        if not report.valid:
            print(
                "VERIFICATION FAILURE in local result receipt. Refusing to report a length.",
                file=sys.stderr,
            )
            raise RuntimeError("local independent verification rejected a Modal result")
        print(
            json.dumps(
                {
                    "repo_head_sha": payload["repo_head_sha"],
                    "seed": payload["seed"],
                    "length": report.actual_length,
                    "verified": True,
                    "artifact_path": payload["artifact_path"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
