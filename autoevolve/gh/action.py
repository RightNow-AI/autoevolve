"""GitHub Actions entrypoint for approved issue-driven evolution."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoevolve.core.types import Budget, Contract
from autoevolve.gh.api import GhApiError, GitHubClient
from autoevolve.gh.comments import (
    approval_declined_comment,
    ceiling_analysis_comment,
    configuration_error_comment,
    milestone_comment,
    terminal_comment,
)
from autoevolve.gh.opened import handle_opened as handle_opened_proposal
from autoevolve.gh.parse import extract_config, extract_goal
from autoevolve.gh.prpost import build_terminal_pr

APPROVAL_LABEL = "evolve:approved"
EVOLVE_LABEL = "evolve"
_ALLOWED_APPROVER_PERMISSIONS = {"admin", "write"}


def main() -> int:
    """Read the GitHub event environment and dispatch one issue-mode event."""

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("[autoevolve-gh] GITHUB_EVENT_PATH is not set")
        return 1
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[autoevolve-gh] could not read event payload: {exc}")
        return 1
    if not isinstance(event, dict):
        print("[autoevolve-gh] event payload must be a JSON object")
        return 1

    if _event_route(event_name, event) is None:
        print(f"[autoevolve-gh] ignored event {event_name or 'unknown'}")
        return 0

    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    home = Path(os.environ.get("AUTOEVOLVE_HOME", Path.home() / ".autoevolve"))
    workdir = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
    try:
        with GitHubClient(token, repository, api_url) as client:
            return dispatch(event_name, event, client, home=home, workdir=workdir)
    except (GhApiError, OSError, RuntimeError, ValueError) as exc:
        print(f"[autoevolve-gh] action failed: {exc}")
        return 1


def dispatch(
    event_name: str,
    event: dict[str, Any],
    client: GitHubClient,
    *,
    home: Path,
    workdir: Path,
) -> int:
    """Dispatch supported issue events and ignore all other events."""

    issue = event.get("issue")
    if not isinstance(issue, dict):
        print("[autoevolve-gh] ignored issues event without an issue payload")
        return 0

    route = _event_route(event_name, event)
    if route == "opened":
        print(f"[autoevolve-gh] preparing proposal for issue {issue.get('number')}")
        handle_opened(client, issue, workdir=workdir)
        print(f"[autoevolve-gh] posted proposal for issue {issue.get('number')}")
        return 0

    if route == "approved":
        actor = _actor_login(event)
        return handle_approved(
            client,
            issue,
            actor=actor,
            home=home,
            workdir=workdir,
        )

    print(f"[autoevolve-gh] ignored issues action {event.get('action') or 'unknown'}")
    return 0


def handle_opened(client: GitHubClient, issue: dict[str, Any], *, workdir: Path) -> None:
    """Bridge the safe proposal module to source-only evaluator synthesis."""

    def synthesize_source(goal: str) -> str:
        with tempfile.TemporaryDirectory(prefix="autoevolve-gh-proposal-") as raw_dir:
            evaluator = _synthesize_evaluator(goal, Path(raw_dir))
            source = evaluator if evaluator.is_file() else evaluator / "evaluate.py"
            if not source.is_file():
                raise ValueError("Synthesis did not produce evaluate.py.")
            return source.read_text(encoding="utf-8")

    handle_opened_proposal(client, issue, synthesize_source)


def handle_approved(
    client: GitHubClient,
    issue: dict[str, Any],
    *,
    actor: str,
    home: Path,
    workdir: Path,
) -> int:
    """Verify approval, lock the measured contract, run, render, and open a PR."""

    issue_number = int(issue["number"])
    print(f"[autoevolve-gh] verifying approval for issue {issue_number}")
    labels = client.list_labels(issue_number)
    if APPROVAL_LABEL not in {_label_name(label) for label in labels}:
        client.post_comment(
            issue_number,
            approval_declined_comment(
                actor,
                f"The issue does not currently have the {APPROVAL_LABEL} label",
            ),
        )
        print(f"[autoevolve-gh] declined approval for issue {issue_number}: label missing")
        return 0

    if actor == "unknown":
        client.post_comment(
            issue_number,
            approval_declined_comment(actor, "The label event did not identify its actor"),
        )
        print(f"[autoevolve-gh] declined approval for issue {issue_number}: actor missing")
        return 0
    try:
        permission_payload = client.get_actor_permission(actor)
    except GhApiError as exc:
        if exc.status not in {403, 404}:
            raise
        permission_payload = {"permission": "none"}
    permission = permission_payload.get("permission")
    if permission not in _ALLOWED_APPROVER_PERMISSIONS:
        reason = "The label actor must have admin or write permission on this repository"
        client.post_comment(issue_number, approval_declined_comment(actor, reason))
        print(f"[autoevolve-gh] declined approval for issue {issue_number}: permission")
        return 0
    print(f"[autoevolve-gh] approval verified for issue {issue_number} by {actor}")

    try:
        goal = extract_goal(str(issue.get("title", "")), issue.get("body"))
        config = extract_config(issue.get("body"))
    except ValueError as exc:
        client.post_comment(issue_number, configuration_error_comment(str(exc)))
        print(f"[autoevolve-gh] rejected invalid configuration for issue {issue_number}")
        return 0

    evaluator_temp: tempfile.TemporaryDirectory[str] | None = None
    run_id: str | None = None
    try:
        evaluator_ref: Path
        configured_evaluator = config.get("evaluator")
        if isinstance(configured_evaluator, str):
            evaluator_ref = _within_workdir(workdir, configured_evaluator)
            if not evaluator_ref.exists():
                raise ValueError(f"Configured evaluator does not exist: {configured_evaluator}")
        else:
            evaluator_temp = tempfile.TemporaryDirectory(prefix="autoevolve-gh-approved-")
            evaluator_ref = _synthesize_evaluator(goal, Path(evaluator_temp.name))
        print(f"[autoevolve-gh] evaluator ready for issue {issue_number}")

        engine = _new_engine(home)
        budget = Budget(
            max_evals=int(config["budget_evals"]),
            wall_clock_s=_optional_float(config.get("wall_clock_s")),
        )
        opened = engine.open_run(
            goal,
            evaluator_ref=evaluator_ref,
            budget=budget,
            workers=int(config["workers"]),
        )
        run_id = str(opened["run_id"])
        contract = opened.get("contract")
        print(f"[autoevolve-gh] baseline measured and contract locked for run {run_id}")

        if _is_infeasible(opened, contract):
            analysis = _ceiling_analysis(opened, contract)
            client.post_comment(issue_number, ceiling_analysis_comment(run_id, analysis))
            print(f"[autoevolve-gh] run {run_id} stopped as infeasible")
            return 0

        cadence = max(10, int(config["budget_evals"]) // 5)
        submission_count = 0

        def on_submission(submission: dict[str, Any] | None = None) -> None:
            nonlocal submission_count
            submission_count += 1
            if submission_count % cadence != 0:
                return
            status = engine.run_status(run_id)
            curve = _curve_rows(status.get("curve"))
            best_fitness = _best_fitness(status, submission, curve)
            artifacts_note = _artifacts_note(status.get("artifacts"))
            client.post_comment(
                issue_number,
                milestone_comment(
                    run_id,
                    submission_count,
                    int(config["budget_evals"]),
                    best_fitness,
                    _baseline(contract),
                    curve,
                    artifacts_note,
                ),
            )
            print(f"[autoevolve-gh] posted milestone {submission_count} for run {run_id}")

        print(f"[autoevolve-gh] starting evolution for run {run_id}")
        loop_result = _run_loop(
            engine,
            run_id,
            workdir=workdir,
            workers=int(config["workers"]),
            evaluator_dir=evaluator_ref,
            on_submission=on_submission,
        )
        status = engine.run_status(run_id)
        if isinstance(loop_result, dict):
            status = {**loop_result, **status}
        print(f"[autoevolve-gh] evolution closed for run {run_id}")

        artifact_dir = _artifact_dir(workdir, run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        rendered = _render_run(home, run_id, artifact_dir)
        report_path = artifact_dir / "report.md"
        report_result = _report_run(home, run_id, report_path)
        _ensure_report_artifact(report_result, report_path)
        _normalize_rendered_artifacts(rendered, artifact_dir)
        summary = _summary(status, report_result, artifact_dir)
        status_name = _status_name(status)
        _stage_pr_inputs(engine, run_id, home, artifact_dir, config, contract, summary)
        print(f"[autoevolve-gh] rendered terminal artifacts for run {run_id}")

        client.post_comment(issue_number, terminal_comment(run_id, status_name, summary))
        print(f"[autoevolve-gh] posted terminal comment for run {run_id}")
        if _opens_pull_request(status_name):
            url = build_terminal_pr(client, run_id, home, artifact_dir)
            print(f"[autoevolve-gh] opened terminal pull request for run {run_id}: {url}")
        return 0
    except Exception as exc:
        if run_id is not None:
            summary = f"The run stopped because {_public_error(exc)}"
            client.post_comment(issue_number, terminal_comment(run_id, "failed", summary))
        print(
            f"[autoevolve-gh] approved run failed for issue {issue_number}: "
            f"{_public_error(exc)}"
        )
        raise
    finally:
        if evaluator_temp is not None:
            evaluator_temp.cleanup()


def _synthesize_evaluator(goal: str, workdir: Path) -> Path:
    from autoevolve.mutate.models import resolve_endpoint
    from autoevolve.synth.pipeline import synthesize

    endpoint = resolve_endpoint("strong") or resolve_endpoint("cheap")
    if endpoint is None:
        raise ValueError("Evaluator synthesis requires a configured model endpoint.")
    return Path(synthesize(goal, workdir, endpoint))


def _new_engine(home: Path) -> Any:
    from autoevolve.core.engine import Engine

    return Engine(home=home)


def _run_loop(
    engine: Any,
    run_id: str,
    *,
    workdir: Path,
    workers: int,
    evaluator_dir: Path | None = None,
    on_submission: Callable[[dict[str, Any] | None], None],
) -> dict[str, Any] | None:
    from autoevolve.core.loop import run_worker_loop
    from autoevolve.mutate.compose import build_get_operator, build_local_evaluator

    return run_worker_loop(
        engine,
        run_id,
        build_get_operator(None, build_local_evaluator(evaluator_dir)),
        on_submission=on_submission,
    )


def _render_run(home: Path, run_id: str, out_dir: Path) -> dict[str, Any]:
    from autoevolve.cli.render import render_all

    return render_all(home, run_id, out_dir, live=False)


def _report_run(home: Path, run_id: str, out_path: Path) -> Any:
    from autoevolve.cli.report import report

    return report(home, run_id, out_path)


def _requests_evolution(issue: dict[str, Any]) -> bool:
    title = str(issue.get("title", "")).lstrip().lower()
    labels = issue.get("labels")
    names = {_label_name(label) for label in labels} if isinstance(labels, list) else set()
    return title.startswith("evolve:") or EVOLVE_LABEL in names


def _event_route(event_name: str, event: dict[str, Any]) -> str | None:
    if event_name != "issues":
        return None
    issue = event.get("issue")
    if not isinstance(issue, dict):
        return None
    event_action = str(event.get("action", ""))
    if event_action in {"opened", "edited"} and _requests_evolution(issue):
        return "opened"
    event_label = event.get("label")
    label_name = event_label.get("name") if isinstance(event_label, dict) else None
    if event_action == "labeled" and label_name == APPROVAL_LABEL:
        return "approved"
    return None


def _label_name(label: object) -> str:
    if isinstance(label, str):
        return label
    if isinstance(label, dict):
        name = label.get("name")
        return name if isinstance(name, str) else ""
    return ""


def _actor_login(event: dict[str, Any]) -> str:
    sender = event.get("sender")
    if isinstance(sender, dict) and isinstance(sender.get("login"), str):
        return sender["login"]
    return "unknown"


def _within_workdir(workdir: Path, relative: str) -> Path:
    root = workdir.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Evaluator path must stay inside the checked-out repository.") from exc
    return candidate


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _public_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    for key, value in os.environ.items():
        upper = key.upper()
        if value and any(marker in upper for marker in ("KEY", "PASSWORD", "SECRET", "TOKEN")):
            message = message.replace(value, "[redacted]")
    detail = message or "no additional detail was provided"
    return f"{type(error).__name__}: {detail}"


def _is_infeasible(opened: dict[str, Any], contract: object) -> bool:
    if opened.get("infeasible") is True or opened.get("status") == "infeasible":
        return True
    feasibility = _contract_field(contract, "feasibility")
    return isinstance(feasibility, dict) and feasibility.get("infeasible") is True


def _ceiling_analysis(opened: dict[str, Any], contract: object) -> str:
    analysis = opened.get("analysis") or opened.get("ceiling_analysis")
    if analysis is None:
        analysis = _contract_field(contract, "feasibility")
    if isinstance(analysis, str):
        return analysis
    if isinstance(analysis, dict):
        ceiling = analysis.get("value", analysis.get("ceiling", "unknown"))
        method = analysis.get("method", "the evaluator ceiling method")
        plausible = analysis.get("maximum_plausible_target", ceiling)
        return (
            f"The requested target exceeds the measured ceiling {ceiling} from {method}. "
            f"The maximum plausible target is {plausible}."
        )
    return "The locked evaluator reported that the requested target is infeasible."


def _baseline(contract: object) -> float | None:
    value = _contract_field(contract, "baseline")
    return float(value) if isinstance(value, int | float) else None


def _contract_field(contract: object, name: str) -> Any:
    if isinstance(contract, dict):
        return contract.get(name)
    return getattr(contract, name, None)


def _curve_rows(value: object) -> list[tuple[int, float]]:
    if not isinstance(value, list):
        return []
    rows: list[tuple[int, float]] = []
    for row in value:
        if isinstance(row, list | tuple) and len(row) >= 2:
            rows.append((int(row[0]), float(row[1])))
    return rows


def _best_fitness(
    status: dict[str, Any],
    submission: dict[str, Any] | None,
    curve: list[tuple[int, float]],
) -> float | None:
    value = status.get("best_fitness")
    if value is None and submission is not None:
        value = submission.get("best_fitness")
    if value is None and curve:
        value = curve[-1][1]
    return float(value) if isinstance(value, int | float) else None


def _artifacts_note(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "Terminal artifacts will be attached when this run closes"
    paths = [str(path) for path in value.values() if path]
    if not paths:
        return "Terminal artifacts will be attached when this run closes"
    return f"Current artifact paths: {', '.join(paths)}"


def _artifact_dir(workdir: Path, run_id: str) -> Path:
    configured = os.environ.get("AUTOEVOLVE_ARTIFACTS_DIR")
    root = Path(configured) if configured else workdir / "autoevolve-runs"
    return root / run_id


def _normalize_rendered_artifacts(rendered: dict[str, Any], out_dir: Path) -> None:
    expected = {
        "gif": out_dir / "evolution.gif",
        "poster_png": out_dir / "lineage_poster.png",
    }
    for key, destination in expected.items():
        raw_source = rendered.get(key)
        if raw_source is None:
            continue
        source = Path(raw_source)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)


def _ensure_report_artifact(report_result: Any, out_path: Path) -> None:
    if out_path.is_file():
        return
    if isinstance(report_result, Path) and report_result.is_file():
        shutil.copyfile(report_result, out_path)
        return
    if isinstance(report_result, str):
        if "\n" not in report_result:
            candidate = Path(report_result)
            if not candidate.is_absolute():
                candidate = out_path.parent / candidate
            try:
                if candidate.is_file():
                    shutil.copyfile(candidate, out_path)
                    return
            except OSError:
                pass
        out_path.write_text(report_result, encoding="utf-8")
        return
    raise ValueError("The report renderer did not produce report.md.")


def _summary(status: dict[str, Any], report_result: Any, artifact_dir: Path) -> str:
    if isinstance(report_result, str) and "\n" in report_result:
        paragraphs = [part.strip() for part in report_result.split("\n\n") if part.strip()]
        for paragraph in paragraphs:
            if not paragraph.startswith("#"):
                return " ".join(paragraph.split())
    report_path = artifact_dir / "report.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in report.split("\n\n") if part.strip()]
        for paragraph in paragraphs:
            if not paragraph.startswith("#"):
                return " ".join(paragraph.split())
    status_name = _status_name(status)
    if status_name == "target_hit":
        return "The run reached its locked target. The pull request contains the measured winner."
    if status_name in {"budget_exhausted", "plateau", "plateau_detected", "best_found"}:
        return "The run closed with the best measured candidate found under the locked contract."
    return f"The run closed with status {status_name}. See report.md for the measured result."


def _status_name(status: dict[str, Any]) -> str:
    value = status.get("status", "best_found")
    return str(value).lower().replace(" ", "_")


def _opens_pull_request(status: str) -> bool:
    return status in {
        "best_found",
        "budget_exhausted",
        "closed",
        "completed",
        "plateau",
        "plateau_detected",
        "target_hit",
    }


def _stage_pr_inputs(
    engine: Any,
    run_id: str,
    home: Path,
    artifact_dir: Path,
    config: dict[str, Any],
    contract: object,
    summary: str,
) -> None:
    winner_dir = artifact_dir / "winner"
    winner_dir.mkdir(parents=True, exist_ok=True)
    best = engine.best(run_id, k=1)
    if best:
        winner = best[0]
        files = winner.get("files") if isinstance(winner, dict) else None
        if isinstance(files, dict):
            for relative, content in files.items():
                destination = _safe_child(winner_dir, str(relative))
                destination.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    destination.write_bytes(content)
                else:
                    destination.write_text(str(content), encoding="utf-8")
        else:
            code_ref = winner.get("code_ref") if isinstance(winner, dict) else None
            if isinstance(code_ref, str):
                source = Path(code_ref)
                if not source.is_absolute():
                    source = home / "store" / source
                _copy_tree(source, winner_dir)

    manifest = {
        "contract": _contract_text(contract),
        "result": summary,
        "target_path": config.get("target_path", f"autoevolve-results/{run_id}"),
    }
    (artifact_dir / ".autoevolve-gh.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Winning candidate path escapes its root: {relative!r}") from exc
    return candidate


def _contract_text(contract: object) -> str:
    if isinstance(contract, Contract):
        data = json.loads(contract.to_json())
    elif isinstance(contract, dict):
        data = contract
    else:
        return str(contract)
    budget = data.get("budget", {})
    if isinstance(budget, Budget):
        budget = {
            "max_evals": budget.max_evals,
            "wall_clock_s": budget.wall_clock_s,
        }
    bounds = []
    if isinstance(budget, dict) and budget.get("max_evals") is not None:
        bounds.append(f"{budget['max_evals']} evaluations")
    if isinstance(budget, dict) and budget.get("wall_clock_s") is not None:
        bounds.append(f"{budget['wall_clock_s']} seconds")
    target = data.get("target")
    return "\n".join(
        [
            "CONTRACT",
            f"goal: {data.get('goal', '')}",
            f"metric: {data.get('metric', '')}  baseline: {data.get('baseline')}  "
            f"target: {target if target is not None else 'maximize'}",
            f"gate: {data.get('gate', '')}   budget: {', '.join(bounds)}",
            f"feasibility: {data.get('feasibility') or 'unbounded; plateau detection governs'}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
