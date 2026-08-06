"""Evaluate one VRPTW solver across a public instance portfolio on Modal.

Usage:
    modal run campaigns/vrp/modal_portfolio.py --solver baseline \
        --family homberger_400 --limit 60 --minutes 30 --store-name portfolio-vrp
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = Path("/root/autoevolve")
EVALUATOR_RELATIVE = Path("campaigns/vrp/evaluators/vrp")
FIXTURES_RELATIVE = EVALUATOR_RELATIVE / "fixtures"
FAMILIES = frozenset(
    {
        "solomon",
        "homberger_200",
        "homberger_400",
        "homberger_600",
        "homberger_800",
        "homberger_1000",
    }
)
_EVALUATE_SCRIPT = """
import json
import sys
from pathlib import Path

from autoevolve.eval.cascade import run_cascade
from autoevolve.eval.contract import load_evaluator

outcome = run_cascade(load_evaluator(Path(sys.argv[1])), Path(sys.argv[2]))
print(json.dumps({
    "gate_passed": outcome.gate_passed,
    "scores": outcome.scores,
    "error": outcome.error,
}))
"""


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
        # The clone command never varies, so its layer can be served from cache
        # holding an older tree while only the checkout line busts. Fetching the
        # pinned commit first makes the checkout independent of the cache age.
        f"cd {REPO_ROOT} && git fetch origin {COMMIT} && git checkout --detach {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
        f"printf '%s' '{COMMIT}' > {REPO_ROOT}/.autoevolve-image-commit",
    )
)

store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
app = modal.App("autoevolve-vrp-portfolio")


def _safe_store_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("store_name must be one plain path component")
    return value


def _validate_family(value: str) -> str:
    if value not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(sorted(FAMILIES))}")
    return value


def _local_instances(family: str, limit: int) -> list[str]:
    family = _validate_family(family)
    if limit < 1:
        raise ValueError("limit must be positive")
    fixtures = Path(__file__).resolve().parents[2] / FIXTURES_RELATIVE
    family_dir = fixtures / family
    if not family_dir.is_dir():
        raise RuntimeError(
            f"fixture family is missing: {family_dir}; run campaigns/vrp/fetch_instances.py"
        )
    paths = sorted(family_dir.glob("*.txt"), key=lambda path: path.name.casefold())
    if not paths:
        raise RuntimeError(f"fixture family contains no .txt instances: {family_dir}")
    return [f"{family}/{path.name}" for path in paths[:limit]]


def _store_solver_path(value: str) -> Path:
    raw = PurePosixPath(value)
    if raw.is_absolute():
        if not raw.parts or raw.parts[0] != "/" or raw.parts[1:2] != ("store",):
            raise ValueError("solver must be baseline or a path inside /store")
        raw = PurePosixPath(*raw.parts[2:])
    if not raw.parts or ".." in raw.parts:
        raise ValueError("solver store path must not be empty or contain '..'")
    root = Path("/store").resolve()
    resolved = (root / Path(*raw.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("solver path escapes /store") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"solver file does not exist: {resolved}")
    return resolved


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=2.0,
    memory=4096,
    timeout=60 * 60 * 24,
    single_use_containers=True,
)
def run_instance(job: dict[str, object]) -> dict[str, object]:
    """Evaluate one solver-instance pair and return failure as data."""

    import json
    import math
    import os
    import shutil
    import subprocess
    import tempfile

    instance = str(job["instance"])
    solver = str(job["solver"])
    timeout_s = float(job["timeout_s"])
    row: dict[str, object] = {
        "instance": instance,
        "gate_passed": False,
        "vehicle_count": None,
        "total_distance": None,
        "error": None,
    }
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        os.environ["AUTOEVOLVE_CELL"] = f"file:{instance}"
        os.environ["AUTOEVOLVE_VRP_TIMEOUT_S"] = str(timeout_s)

        evaluator_dir = REPO_ROOT / EVALUATOR_RELATIVE

        def evaluate_candidate(candidate_dir: Path) -> dict[str, object]:
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "python",
                    "-c",
                    _EVALUATE_SCRIPT,
                    str(evaluator_dir),
                    str(candidate_dir),
                ],
                cwd=REPO_ROOT,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_s + 60.0,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "")[-2000:].strip()
                raise RuntimeError(
                    f"evaluator process exited {completed.returncode}"
                    + (f": {stderr}" if stderr else "")
                )
            lines = (completed.stdout or "").splitlines()
            if not lines:
                raise RuntimeError("evaluator process returned no result")
            payload = json.loads(lines[-1])
            if not isinstance(payload, dict):
                raise RuntimeError("evaluator process result is not an object")
            return payload

        if solver == "baseline":
            candidate_dir = evaluator_dir / "baseline"
            outcome = evaluate_candidate(candidate_dir)
        else:
            store.reload()
            source = _store_solver_path(solver)
            with tempfile.TemporaryDirectory(prefix="vrp-solver-") as temporary:
                candidate_dir = Path(temporary)
                shutil.copy2(source, candidate_dir / "solver.py")
                outcome = evaluate_candidate(candidate_dir)

        if outcome.get("gate_passed") is not True:
            error = outcome.get("error")
            row["error"] = str(error) if error else "evaluator gate failed without a reason"
            return row
        scores = outcome.get("scores")
        if not isinstance(scores, dict):
            raise RuntimeError("gate-passing evaluator result omitted scores")
        vehicle_count = scores.get("vehicle_count")
        total_distance = scores.get("total_distance")
        if vehicle_count is None or total_distance is None:
            raise RuntimeError("gate-passing evaluator result omitted objective metrics")
        integer_vehicles = int(vehicle_count)
        if float(integer_vehicles) != float(vehicle_count):
            raise RuntimeError("evaluator returned a non-integral vehicle count")
        row.update(
            {
                "gate_passed": True,
                "vehicle_count": integer_vehicles,
                "total_distance": float(total_distance),
            }
        )
    except Exception as exc:  # noqa: BLE001 - each portfolio cell must survive independently
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _objective_module():
    """Load campaigns/vrp/objective.py by path, once per container.

    The campaigns tree ships no __init__.py, so `from campaigns.vrp.objective
    import ...` depends on namespace package resolution and raises
    ModuleNotFoundError inside the container. It also ran in functions where
    sys.path had never been extended, which is how a completed twenty instance
    sweep was lost at the final persist step.
    """

    import importlib.util
    import sys

    cached = sys.modules.get("vrp_objective")
    if cached is not None:
        return cached
    path = REPO_ROOT / "campaigns" / "vrp" / "objective.py"
    spec = importlib.util.spec_from_file_location("vrp_objective", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the vrp objective module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrp_objective"] = module
    spec.loader.exec_module(module)
    return module


def _load_bounds() -> dict[str, dict[str, object]]:
    import json

    objective = _objective_module()
    BOUND_CLAIM_PREFIX = objective.BOUND_CLAIM_PREFIX
    decode_objective_value = objective.decode_objective_value

    path = REPO_ROOT / "campaigns" / "vrp" / "bounds.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, dict[str, object]] = {}
    for entry in payload.get("bounds", []):
        if not isinstance(entry, dict):
            continue
        claim = entry.get("claim")
        value = entry.get("value")
        if not isinstance(claim, str) or not claim.startswith(BOUND_CLAIM_PREFIX):
            continue
        if not isinstance(value, str):
            continue
        try:
            objective = decode_objective_value(value)
        except ValueError:
            continue
        instance = claim.removeprefix(BOUND_CLAIM_PREFIX).casefold()
        index[instance] = {"objective": objective, "entry": entry}
    return index


def _annotate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    is_better_result = _objective_module().is_better_result

    bounds = _load_bounds()
    annotated: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        instance_key = Path(str(row["instance"])).stem.casefold()
        bound = bounds.get(instance_key)
        row["comparison"] = "no_bound"
        row["bound"] = None
        if bound is not None:
            row["bound"] = bound["entry"]
            if bool(row.get("gate_passed")):
                candidate = (int(row["vehicle_count"]), float(row["total_distance"]))
                incumbent = bound["objective"]
                if is_better_result(candidate, incumbent):
                    row["comparison"] = "apparent_improvement_requires_source_recheck"
                elif is_better_result(incumbent, candidate):
                    row["comparison"] = "behind_best_known"
                else:
                    row["comparison"] = "matches_best_known_within_epsilon"
            else:
                row["comparison"] = "gate_failed"
        annotated.append(row)
    return annotated


def _readable_summary(payload: dict[str, object]) -> str:
    rows = payload["results"]
    assert isinstance(rows, list)
    lines = [
        f"VRP portfolio: {payload['family']}",
        f"solver: {payload['solver']}",
        f"minutes per instance: {payload['minutes']}",
        f"image commit: {payload['image_commit']}",
        "",
        "instance | gate | vehicles | distance | comparison | error",
        "--- | --- | ---: | ---: | --- | ---",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            " | ".join(
                (
                    str(row["instance"]),
                    "pass" if row["gate_passed"] else "fail",
                    str(row["vehicle_count"] if row["vehicle_count"] is not None else ""),
                    str(row["total_distance"] if row["total_distance"] is not None else ""),
                    str(row["comparison"]),
                    str(row["error"] or "").replace("|", "/"),
                )
            )
        )
    return "\n".join(lines) + "\n"


@app.function(image=image, volumes={"/store": store}, timeout=600)
def persist_results(payload: dict[str, object]) -> dict[str, object]:
    """Persist the complete portfolio table and a readable companion summary."""

    import json
    import sys
    from datetime import UTC, datetime

    sys.path.insert(0, str(REPO_ROOT))
    store_name = _safe_store_name(str(payload["store_name"]))
    family = _validate_family(str(payload["family"]))
    store.reload()
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("results must be a list")
    payload = dict(payload)
    payload["results"] = _annotate_rows(rows)
    passed = sum(bool(row.get("gate_passed")) for row in payload["results"])
    payload["summary"] = {
        "total": len(payload["results"]),
        "gate_passed": passed,
        "failed": len(payload["results"]) - passed,
        "apparent_improvements": sum(
            row.get("comparison") == "apparent_improvement_requires_source_recheck"
            for row in payload["results"]
        ),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("/store") / store_name / "vrp-portfolio"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family}-{timestamp}"
    json_path = output_dir / f"{stem}.json"
    text_path = output_dir / f"{stem}.txt"
    latest_json = output_dir / "latest.json"
    latest_text = output_dir / "latest.txt"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    readable = _readable_summary(payload)
    try:
        json_path.write_text(encoded, encoding="utf-8")
        text_path.write_text(readable, encoding="utf-8")
        latest_json.write_text(encoded, encoding="utf-8")
        latest_text.write_text(readable, encoding="utf-8")
    finally:
        store.commit()
    return {
        "json_path": str(json_path),
        "summary_path": str(text_path),
        "latest_json_path": str(latest_json),
        "latest_summary_path": str(latest_text),
        "summary": payload["summary"],
    }


@app.local_entrypoint()
def main(
    solver: str,
    family: str = "homberger_400",
    limit: int = 60,
    minutes: float = 30.0,
    store_name: str = "portfolio-vrp",
) -> None:
    """Map one solver across a bounded fixture family and persist every result."""

    import json
    import math
    from datetime import UTC, datetime

    if solver != "baseline" and not solver.strip():
        raise ValueError("solver must be baseline or a store path")
    if not math.isfinite(minutes) or minutes <= 0.0:
        raise ValueError("minutes must be positive")
    _safe_store_name(store_name)
    instances = _local_instances(family, limit)
    jobs = [
        {"instance": instance, "solver": solver, "timeout_s": minutes * 60.0}
        for instance in instances
    ]

    results: list[dict[str, object]] = []
    mapped = run_instance.map(jobs, return_exceptions=True)
    for job, result in zip(jobs, mapped, strict=True):
        if isinstance(result, BaseException):
            result = {
                "instance": job["instance"],
                "gate_passed": False,
                "vehicle_count": None,
                "total_distance": None,
                "error": f"container failure: {type(result).__name__}: {result}",
            }
        elif not isinstance(result, dict):
            result = {
                "instance": job["instance"],
                "gate_passed": False,
                "vehicle_count": None,
                "total_distance": None,
                "error": f"container returned {type(result).__name__}, expected an object",
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "image_commit": COMMIT,
        "solver": solver,
        "family": family,
        "limit": limit,
        "minutes": minutes,
        "store_name": store_name,
        "results": results,
    }
    persisted = persist_results.remote(payload)
    print(json.dumps(persisted, indent=2, sort_keys=True), flush=True)
