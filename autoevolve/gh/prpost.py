"""Build the terminal pull request for a completed approved run."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any

from autoevolve.gh.api import FileContent, GitHubClient

_ARTIFACT_NAMES = ("report.md", "evolution.gif", "lineage_poster.png")
_MANIFEST_NAME = ".autoevolve-gh.json"


def build_terminal_pr(
    client: GitHubClient,
    run_id: str,
    home: Path,
    workdir: Path,
) -> str:
    """Commit the winner and run artifacts, then open the terminal pull request."""

    manifest = _load_manifest(workdir)
    result_dir = PurePosixPath("autoevolve-results") / run_id
    target_path = _target_path(manifest, home, run_id, result_dir)
    files: dict[str, FileContent] = {}

    for relative, content in _winning_files(home, workdir, run_id).items():
        destination = (target_path / PurePosixPath(relative)).as_posix()
        files[destination] = content

    artifact_paths: dict[str, str] = {}
    for name in _ARTIFACT_NAMES:
        source = _find_artifact(workdir, run_id, name)
        destination = (result_dir / name).as_posix()
        if destination in files:
            raise ValueError(f"Winning candidate collides with required artifact {destination!r}.")
        files[destination] = source.read_bytes()
        artifact_paths[name] = destination

    contract = str(manifest.get("contract") or _contract_from_store(home, run_id))
    result = str(manifest.get("result") or _result_from_report(workdir, run_id))
    body = _pull_request_body(run_id, contract, result, artifact_paths)

    default_branch = client.get_default_branch()
    branch_name = _required_text(default_branch, "name")
    base_sha = _required_text(default_branch, "sha")
    head = f"autoevolve/run-{run_id}"
    client.create_branch(base_sha, head)
    client.put_files(head, files, f"Add AutoEvolve result for {run_id}")
    pull_request = client.create_pr(
        head,
        branch_name,
        f"AutoEvolve result for {run_id}",
        body,
    )
    return _required_text(pull_request, "html_url")


def _load_manifest(workdir: Path) -> dict[str, Any]:
    path = workdir / _MANIFEST_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{_MANIFEST_NAME} must contain a JSON object.")
    return payload


def _target_path(
    manifest: dict[str, Any],
    home: Path,
    run_id: str,
    default: PurePosixPath,
) -> PurePosixPath:
    configured = manifest.get("target_path") or _target_path_from_evaluator(home, run_id)
    if configured is None:
        return default
    raw = str(configured).replace("\\", "/")
    if raw.startswith("/") or (len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":"):
        raise ValueError("target_path must be a repository-relative directory.")
    normalized = raw.strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("target_path must be a repository-relative directory.")
    return path


def _target_path_from_evaluator(home: Path, run_id: str) -> str | None:
    database = home / "autoevolve.db"
    if not database.is_file():
        return None
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT evaluator_ref FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if row is None or not row[0]:
        return None
    evaluator = Path(str(row[0]))
    for name in ("autoevolve.json", _MANIFEST_NAME):
        config_path = evaluator / name
        if not config_path.is_file():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("target_path"), str):
            return payload["target_path"]
    return None


def _winning_files(home: Path, workdir: Path, run_id: str) -> dict[str, bytes]:
    for directory in (workdir / "winner", workdir / run_id / "winner"):
        if directory.is_dir():
            files = _read_tree(directory)
            if files:
                return files

    code_directory = _winning_code_directory(home, run_id)
    files = _read_tree(code_directory)
    if not files:
        raise ValueError(f"No winning candidate files were found for run {run_id}.")
    return files


def _winning_code_directory(home: Path, run_id: str) -> Path:
    database = home / "autoevolve.db"
    if not database.is_file():
        raise FileNotFoundError(f"AutoEvolve database not found at {database}.")
    with sqlite3.connect(database) as connection:
        run = connection.execute(
            "SELECT contract_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"Run {run_id} does not exist in the AutoEvolve store.")
        contract = json.loads(str(run[0]))
        metric = str(contract["metric"])
        direction = "DESC" if bool(contract.get("maximize", True)) else "ASC"
        query = f"""
            SELECT p.code_ref
            FROM programs AS p
            JOIN scores AS s ON s.program_id = p.id
            WHERE p.run_id = ? AND s.metric = ?
            ORDER BY s.stage DESC, s.value {direction}, p.created_at ASC
            LIMIT 1
        """
        winner = connection.execute(query, (run_id, metric)).fetchone()
    if winner is None:
        raise ValueError(f"Run {run_id} has no scored candidate.")
    code_ref = Path(str(winner[0]))
    return code_ref if code_ref.is_absolute() else home / "store" / code_ref


def _read_tree(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _find_artifact(workdir: Path, run_id: str, name: str) -> Path:
    candidates = (
        workdir / name,
        workdir / run_id / name,
        workdir / "autoevolve-runs" / run_id / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required run artifact {name!r} was not found for {run_id}.")


def _contract_from_store(home: Path, run_id: str) -> str:
    database = home / "autoevolve.db"
    if not database.is_file():
        raise FileNotFoundError("A contract manifest or AutoEvolve database is required.")
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT contract_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Run {run_id} has no locked contract.")
    contract = json.loads(str(row[0]))
    budget = contract.get("budget", {})
    budget_parts = []
    if budget.get("max_evals") is not None:
        budget_parts.append(f"{budget['max_evals']} evaluations")
    if budget.get("wall_clock_s") is not None:
        budget_parts.append(f"{budget['wall_clock_s']} seconds")
    target = contract.get("target")
    return "\n".join(
        [
            "CONTRACT",
            f"goal: {contract.get('goal', '')}",
            f"metric: {contract.get('metric', '')}  baseline: {contract.get('baseline')}  "
            f"target: {target if target is not None else 'maximize'}",
            f"gate: {contract.get('gate', '')}   budget: {', '.join(budget_parts)}",
            f"feasibility: {contract.get('feasibility') or 'unbounded; plateau detection governs'}",
        ]
    )


def _result_from_report(workdir: Path, run_id: str) -> str:
    report = _find_artifact(workdir, run_id, "report.md").read_text(encoding="utf-8")
    paragraphs = [part.strip() for part in report.split("\n\n") if part.strip()]
    for paragraph in paragraphs:
        if not paragraph.startswith("#"):
            return " ".join(paragraph.split())
    return "The run report is included in this pull request."


def _pull_request_body(
    run_id: str,
    contract: str,
    result: str,
    artifacts: dict[str, str],
) -> str:
    clean_contract = contract.replace("\N{EM DASH}", "-").strip()
    clean_result = " ".join(result.replace("\N{EM DASH}", "-").split())
    return "\n".join(
        [
            "## Locked contract",
            "",
            "```text",
            clean_contract,
            "```",
            "",
            "## Result",
            "",
            clean_result,
            "",
            f"![Lineage poster]({artifacts['lineage_poster.png']})",
            "",
            f"![Evolution]({artifacts['evolution.gif']})",
            "",
            f"Run id: {run_id}",
        ]
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected {key!r} in GitHub response.")
    return value
