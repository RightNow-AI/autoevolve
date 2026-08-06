"""Fetch SINTEF VRPTW best known tables through Modal and write bounds.json.

Usage:
    modal run campaigns/vrp/fetch_bounds.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = Path("/root/autoevolve")
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
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout --detach {COMMIT}",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)


@app.function(image=image, timeout=60 * 10, cpu=1.0, memory=1024)
def fetch_pages() -> dict[str, object]:
    """Fetch all pages, retaining successful rows when another page fails."""

    import importlib.util
    import urllib.request
    from datetime import UTC, datetime

    # Loaded by path rather than imported as campaigns.vrp.bounds_parser. The
    # campaigns tree ships no __init__.py, so the dotted import depends on
    # namespace package resolution inside the container and fails there.
    parser_path = REPO_ROOT / "campaigns" / "vrp" / "bounds_parser.py"
    if not parser_path.is_file():
        raise RuntimeError(f"bounds parser missing from the image at {parser_path}")
    parser_spec = importlib.util.spec_from_file_location("vrp_bounds_parser", parser_path)
    if parser_spec is None or parser_spec.loader is None:
        raise RuntimeError(f"could not load the bounds parser at {parser_path}")
    parser_module = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser_module)
    parse_sintef_page = parser_module.parse_sintef_page

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
