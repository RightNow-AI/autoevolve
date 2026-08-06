"""Fetch SINTEF VRPTW best known tables through Modal and write bounds.json.

Usage:
    modal run campaigns/vrp/fetch_bounds.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
# The container path stays a plain string wherever it reaches a shell. As a
# Path it stringifies to "\root\autoevolve" on Windows, so the clone landed in a
# directory of that literal name while the runtime code read /root/autoevolve
# and found nothing there.
REMOTE_ROOT = "/root/autoevolve"
REPO_ROOT = Path(REMOTE_ROOT)
PAGES = (
    (
        "solomon_100",
        "https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/",
        56,
    ),
    (
        "homberger_200",
        "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/200-customers/",
        60,
    ),
    (
        "homberger_400",
        "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/400-customers/",
        60,
    ),
    (
        "homberger_600",
        "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/600-customers/",
        60,
    ),
    (
        "homberger_800",
        "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/800-customers/",
        60,
    ),
    (
        "homberger_1000",
        "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/1000-customers/",
        60,
    ),
)


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

app = modal.App("autoevolve-vrp-fetch-bounds")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ca-certificates")
    .run_commands(
        f"git clone {REPO} {REMOTE_ROOT}",
        # The clone command string never varies, so its layer can be served
        # from cache long after the commit it holds went stale. Fetching the
        # exact commit first is what makes the checkout below reliable.
        f"cd {REMOTE_ROOT} && git fetch origin {COMMIT} && git checkout --detach {COMMIT}",
        f"printf '%s' '{COMMIT}' > {REMOTE_ROOT}/.autoevolve-image-commit",
    )
)


@app.function(image=image, timeout=60 * 10, cpu=1.0, memory=1024)
def fetch_pages() -> dict[str, object]:
    """Fetch all pages, retaining successful rows when another page fails."""

    import sys
    import urllib.request
    from datetime import UTC, datetime

    # Imported by dotted name rather than loaded by path, because bounds_parser
    # itself imports campaigns.vrp.objective. Loading the parser by path leaves
    # that inner import unresolvable. With the repository root on sys.path the
    # campaigns tree resolves as a namespace package and both imports work.
    parser_path = REPO_ROOT / "campaigns" / "vrp" / "bounds_parser.py"
    if not parser_path.is_file():
        raise RuntimeError(f"bounds parser missing from the image at {parser_path}")
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    from campaigns.vrp.bounds_parser import parse_sintef_page

    checked_on = datetime.now(UTC).date().isoformat()
    bounds: list[dict[str, str]] = []
    pages: list[dict[str, object]] = []
    for label, requested_url, expected_rows in PAGES:
        page: dict[str, object] = {
            "page": label,
            "requested_url": requested_url,
            "source_url": None,
            "status": "failed",
            "rows": 0,
            "expected_rows": expected_rows,
            "row_errors": [],
            "error": None,
        }
        try:
            request = urllib.request.Request(
                requested_url,
                headers={"User-Agent": "autoevolve-vrp-bounds/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                source_url = response.geturl()
                charset = response.headers.get_content_charset() or "utf-8"
                html = response.read(8_000_000).decode(charset, errors="replace")
            parsed = parse_sintef_page(html, source_url, checked_on)
            page["source_url"] = source_url
            page["rows"] = len(parsed.bounds)
            page["row_errors"] = list(parsed.row_errors)
            if not parsed.bounds:
                page["error"] = "no recognizable complete instance rows"
            else:
                bounds.extend(parsed.bounds)
                complete = len(parsed.bounds) == expected_rows and not parsed.row_errors
                page["status"] = "clean" if complete else "partial"
                if len(parsed.bounds) != expected_rows:
                    page["error"] = (
                        f"parsed {len(parsed.bounds)} of {expected_rows} expected instance rows"
                    )
        except Exception as exc:  # noqa: BLE001 - one bad page must not discard the others
            page["error"] = f"{type(exc).__name__}: {exc}"
        pages.append(page)
    bounds.sort(key=lambda entry: entry["claim"].casefold())
    return {"bounds": bounds, "pages": pages, "checked_on": checked_on}


def _default_destination() -> Path:
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError as exc:
        raise RuntimeError("could not resolve the local repository root") from exc
    return repo_root / "campaigns" / "vrp" / "bounds.json"


@app.local_entrypoint()
def main(destination: str | None = None) -> None:
    """Fetch remotely, write every real row obtained, and report page failures."""

    import json

    result = fetch_pages.remote()
    target = Path(destination).resolve() if destination else _default_destination()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"bounds": result["bounds"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(result['bounds'])} bounds to {target}", flush=True)
    for page in result["pages"]:
        print(json.dumps(page, sort_keys=True), flush=True)
