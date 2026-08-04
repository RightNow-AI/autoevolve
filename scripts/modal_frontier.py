"""Long-running autoevolve frontier search on Modal.

Design note. The store is SQLite. Spreading workers across containers would
put concurrent writers on a network filesystem, where SQLite locking is
unreliable, so this runs ONE container with many threads instead. That uses
the process write lock the engine already has, and it costs nothing in
throughput because a worker spends nearly all its wall clock waiting on a
model call rather than on CPU.

GPU is deliberately absent. Superpermutation and Golomb search are integer
CPU work; a GPU would be idle money. Kernel work is a separate app.

Usage:
    modal run scripts/modal_frontier.py::search
        --evaluator campaigns/golomb-ruler/evaluators/golomb
        --goal "..." --cell order-29 --budget 4000 --parallel 24 --hours 6
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"


def _head_sha() -> str:
    """The commit this image must contain.

    Modal caches images by their build steps. A plain `git clone` of the
    default branch produces an identical step every time, so the cached layer
    is reused and containers keep running whatever code existed at the first
    build. Baking the commit into the step makes the image rebuild whenever
    the repository moves, which is the difference between testing your fixes
    and testing a snapshot from hours ago.
    """

    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


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
app = modal.App("autoevolve-frontier")


@app.function(
    image=image,
    volumes={"/store": store},
    cpu=8.0,
    memory=16384,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def search(
    evaluator: str,
    goal: str,
    budget: int = 2000,
    parallel: int = 24,
    cell: str | None = None,
    seed: int = 1,
    hours: float = 6.0,
    target: float | None = None,
    operators: str = "diff,rewrite,crossover",
    store_name: str = "default",
) -> dict:
    """Run one long frontier search and leave the store on the volume."""

    import os
    import subprocess

    # Each problem gets its own store. Several searches can then run at once
    # without putting concurrent writers on one SQLite file over a network
    # filesystem, where locking is unreliable and corruption would take the
    # database holding every result with it.
    env = dict(os.environ)
    env["AUTOEVOLVE_HOME"] = f"/store/{store_name}/autoevolve"
    env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"/store/{store_name}/runs"
    if cell:
        env["AUTOEVOLVE_CELL"] = cell

    command = [
        "uv",
        "run",
        "autoevolve",
        "run",
        "--evaluator",
        evaluator,
        "--goal",
        goal,
        "--budget-evals",
        str(budget),
        "--wall-clock-s",
        str(int(hours * 3600)),
        "--workers",
        str(parallel),
        "--parallel",
        str(parallel),
        # agentic is excluded by default: it shells out to a coding CLI that
        # is not installed in this image, so every cycle would be a skip.
        "--operators",
        operators,
        "--seed",
        str(seed),
    ]
    if target is not None:
        command += ["--target", str(target)]

    # Commit the volume periodically rather than only at the end. A long run
    # that commits once gives no visibility while it works and loses every
    # result if the container dies before finishing.
    import threading

    finished = threading.Event()

    def checkpoint() -> None:
        while not finished.wait(120.0):
            try:
                store.commit()
            except Exception as exc:  # noqa: BLE001 - checkpointing is best effort
                print(f"checkpoint failed: {exc}", flush=True)

    keeper = threading.Thread(target=checkpoint, daemon=True)
    keeper.start()
    try:
        completed = subprocess.run(
            command,
            cwd="/root/autoevolve",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        finished.set()
        keeper.join(timeout=10)
    store.commit()
    tail = (completed.stdout or "")[-4000:]
    # Printed inside the container so Modal streams it; return values are not
    # surfaced by `modal run`.
    print("=== autoevolve stdout tail ===", flush=True)
    print(tail, flush=True)
    if completed.returncode != 0:
        print("=== stderr tail ===", flush=True)
        print((completed.stderr or "")[-2000:], flush=True)
    print(f"=== exit {completed.returncode} ===", flush=True)
    return {
        "returncode": completed.returncode,
        "stdout_tail": tail,
        "stderr_tail": (completed.stderr or "")[-2000:],
    }


# torch must land in the PROJECT venv, not the system interpreter. The run
# executes through `uv run`, so a plain pip_install is invisible to it and the
# evaluator silently falls back to mock mode, which is what happened on the
# first GPU attempt.
gpu_image = image.run_commands(
    "cd /root/autoevolve && uv pip install torch "
    "--index-url https://download.pytorch.org/whl/cu128"
)


@app.function(
    image=gpu_image,
    volumes={"/store": store},
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("autoevolve-model")],
)
def search_gpu(
    goal: str,
    budget: int = 400,
    parallel: int = 6,
    seed: int = 1,
    hours: float = 3.0,
    store_name: str = "kernel",
) -> dict:
    """Evolve a GPU kernel where the answer cannot be recalled.

    Nobody has published the optimal kernel for this device on this workload,
    so a passing result came from measurement rather than memory. Throughput
    is measured on the device the candidate ran on, against a roofline derived
    from that device's own memory bandwidth.

    Parallelism is deliberately low: every candidate contends for one GPU, so
    threads past a handful only add measurement noise.
    """

    import os
    import subprocess

    env = dict(os.environ)
    env["AUTOEVOLVE_HOME"] = f"/store/{store_name}/autoevolve"
    env["AUTOEVOLVE_ARTIFACTS_DIR"] = f"/store/{store_name}/runs"

    probe = subprocess.run(
        ["uv", "run", "python", "-c", "import torch;print(torch.cuda.get_device_name(0))"],
        cwd="/root/autoevolve",
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"device: {probe.stdout.strip() or probe.stderr.strip()[:200]}", flush=True)

    completed = subprocess.run(
        [
            "uv", "run", "autoevolve", "run",
            "--evaluator", "evaluators/triton-kernel",
            "--goal", goal,
            "--budget-evals", str(budget),
            "--wall-clock-s", str(int(hours * 3600)),
            "--workers", str(parallel),
            "--parallel", str(parallel),
            "--operators", "diff,rewrite",
            "--seed", str(seed),
        ],
        cwd="/root/autoevolve",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    store.commit()
    print((completed.stdout or "")[-4000:], flush=True)
    if completed.returncode != 0:
        print((completed.stderr or "")[-2000:], flush=True)
    return {"returncode": completed.returncode}


@app.function(image=image, volumes={"/store": store}, timeout=900)
def status_all() -> dict:
    """Report every problem store in one container instead of one each."""

    import json as _json
    import shutil
    import sqlite3
    import tempfile
    from pathlib import Path

    store.reload()
    report: dict[str, dict] = {}
    roots = [Path("/store")] + sorted(p for p in Path("/store").iterdir() if p.is_dir())
    scratch = Path(tempfile.mkdtemp())
    for root in roots:
        db_path = root / "autoevolve" / "autoevolve.db"
        name = "legacy" if root == Path("/store") else root.name
        if not db_path.is_file():
            # Say so rather than skipping. A store that vanishes from this
            # report reads as "not launched" when it may be "launched and
            # broken", and that ambiguity has cost this project whole runs.
            report[name] = {"error": "no database at " + str(db_path)}
            continue
        try:
            # Copy the database and its write-ahead log somewhere writable,
            # then open it normally. A read-only connection cannot build the
            # shared-memory index a WAL needs, so it reports the last
            # checkpoint and a live run looks like an empty one.
            local = scratch / f"{name}.db"
            shutil.copyfile(db_path, local)
            for suffix in ("-wal", "-shm"):
                side = db_path.with_name(db_path.name + suffix)
                if side.is_file():
                    shutil.copyfile(side, local.with_name(local.name + suffix))
            conn = sqlite3.connect(local)
            rows = conn.execute(
                "SELECT id, status, contract_json FROM runs ORDER BY created_at DESC LIMIT 1"
            ).fetchall()
            if not rows:
                report[name] = {"error": "database present but no run was ever opened"}
                continue
            run_id, status, contract = rows[0]
            spec = _json.loads(contract)
            metric, gate = spec["metric"], spec["gate"]
            order = "DESC" if spec["maximize"] else "ASC"
            best_row = conn.execute(
                f"SELECT s.value FROM scores s JOIN programs p ON p.id = s.program_id "
                f"JOIN scores g ON g.program_id = p.id AND g.stage = s.stage "
                f"WHERE p.run_id = ? AND s.metric = ? AND g.metric = ? AND g.value = 1.0 "
                f"ORDER BY s.value {order} LIMIT 1",
                (run_id, metric, gate),
            ).fetchone()
            programs = conn.execute(
                "SELECT COUNT(*) FROM programs WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            mix = dict(
                conn.execute(
                    "SELECT operator, COUNT(*) FROM programs WHERE run_id = ? GROUP BY operator",
                    (run_id,),
                ).fetchall()
            )
            conn.close()
            report[name] = {
                "run_id": run_id,
                "status": status,
                "metric": metric,
                "best": best_row[0] if best_row else None,
                "programs": programs,
                "operators": mix,
            }
        except sqlite3.Error as exc:
            report[name] = {"error": str(exc)}
    print(_json.dumps(report, indent=2), flush=True)
    return report


@app.function(image=image, volumes={"/store": store}, timeout=900)
def verify(store_name: str = "default") -> dict:
    """Report the best result at each stage, and the source of the best one.

    Stages are not interchangeable evidence. A frontier pack screens cheaply at
    stage 0 and proves at its last stage, where a Ramsey certificate is
    re-derived in a fresh interpreter and checked by an exhaustive verifier
    that must agree with two independent fast ones. A number that only ever
    reached stage 0 is a candidate, not a result, and reporting the two the
    same way is how a project talks itself into a claim it cannot defend.
    """

    import json as _json
    import shutil
    import sqlite3
    import tempfile
    from pathlib import Path

    store.reload()
    db_path = Path(f"/store/{store_name}/autoevolve/autoevolve.db")
    if not db_path.is_file():
        return {"error": f"no store at {db_path}"}
    scratch = Path(tempfile.mkdtemp())
    local = scratch / "read.db"
    shutil.copyfile(db_path, local)
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.is_file():
            shutil.copyfile(side, local.with_name(local.name + suffix))
    conn = sqlite3.connect(local)
    row = conn.execute("SELECT id, contract_json FROM runs ORDER BY created_at DESC LIMIT 1")
    found = row.fetchone()
    if found is None:
        return {"error": "no runs"}
    run_id, contract = found
    spec = _json.loads(contract)
    metric, gate = spec["metric"], spec["gate"]
    order = "DESC" if spec["maximize"] else "ASC"

    per_stage: dict[str, dict] = {}
    stages = [
        value
        for (value,) in conn.execute(
            "SELECT DISTINCT stage FROM scores s JOIN programs p ON p.id = s.program_id "
            "WHERE p.run_id = ? ORDER BY stage",
            (run_id,),
        ).fetchall()
    ]
    for stage in stages:
        best_row = conn.execute(
            f"SELECT s.value, p.id, p.operator FROM scores s "
            f"JOIN programs p ON p.id = s.program_id "
            f"JOIN scores g ON g.program_id = p.id AND g.stage = s.stage "
            f"WHERE p.run_id = ? AND s.stage = ? AND s.metric = ? AND g.metric = ? "
            f"AND g.value = 1.0 ORDER BY s.value {order} LIMIT 1",
            (run_id, stage, metric, gate),
        ).fetchone()
        gated = conn.execute(
            "SELECT COUNT(*) FROM scores s JOIN programs p ON p.id = s.program_id "
            "WHERE p.run_id = ? AND s.stage = ? AND s.metric = ? AND s.value = 1.0",
            (run_id, stage, gate),
        ).fetchone()[0]
        per_stage[f"stage_{stage}"] = {
            "best": best_row[0] if best_row else None,
            "program_id": best_row[1] if best_row else None,
            "operator": best_row[2] if best_row else None,
            "gate_passed_count": gated,
        }
    report = {
        "run_id": run_id,
        "metric": metric,
        "gate": gate,
        "last_stage": max(stages) if stages else None,
        "per_stage": per_stage,
    }
    print(_json.dumps(report, indent=2), flush=True)
    return report


@app.function(image=image, volumes={"/store": store}, timeout=900)
def best(run_id: str | None = None, store_name: str = "default") -> dict:
    """Report the best measured result on one problem store without a rerun."""

    import sqlite3
    from pathlib import Path

    store.reload()
    db_path = Path(f"/store/{store_name}/autoevolve/autoevolve.db")
    if not db_path.is_file():
        # Runs started before stores were namespaced live at the volume root.
        legacy = Path("/store/autoevolve/autoevolve.db")
        if legacy.is_file():
            db_path = legacy
        else:
            print(f"no store at {db_path}", flush=True)
            return {"error": f"no store at {db_path}"}
    conn = sqlite3.connect(db_path)
    if run_id is None:
        row = conn.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return {"error": "no runs"}
        run_id = row[0]
    status, contract = conn.execute(
        "SELECT status, contract_json FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    import json as _json

    metric = _json.loads(contract)["metric"]
    maximize = _json.loads(contract)["maximize"]
    order = "DESC" if maximize else "ASC"
    gate = _json.loads(contract)["gate"]
    # Gate failures record the metric as 0.0. For a minimized metric that
    # sorts to the front, so the best candidate must be filtered to those
    # that actually passed the gate at the same stage.
    best_row = conn.execute(
        f"SELECT s.value, p.id, p.code_ref FROM scores s "
        f"JOIN programs p ON p.id = s.program_id "
        f"JOIN scores g ON g.program_id = p.id AND g.stage = s.stage "
        f"WHERE p.run_id = ? AND s.metric = ? AND g.metric = ? AND g.value = 1.0 "
        f"ORDER BY s.value {order} LIMIT 1",
        (run_id, metric, gate),
    ).fetchone()
    passed = conn.execute(
        "SELECT COUNT(DISTINCT s.program_id) FROM scores s JOIN programs p "
        "ON p.id = s.program_id WHERE p.run_id = ? AND s.metric = ? AND s.value = 1.0",
        (run_id, gate),
    ).fetchone()[0]
    operator_mix = dict(
        conn.execute(
            "SELECT operator, COUNT(*) FROM programs WHERE run_id = ? GROUP BY operator",
            (run_id,),
        ).fetchall()
    )
    arms = {
        row[0]: {"pulls": row[1], "improvements": row[2], "mean_gain": round(row[3], 4)}
        for row in conn.execute(
            "SELECT name, pulls, improvements, mean_gain FROM operators WHERE domain = ?",
            (_json.loads(contract)["domain"],),
        )
    }
    plateaus = conn.execute(
        "SELECT COUNT(*) FROM events WHERE run_id = ? AND kind = 'plateau_detected'",
        (run_id,),
    ).fetchone()[0]
    count = conn.execute(
        "SELECT COUNT(*) FROM programs WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    code = ""
    if best_row is not None:
        source = Path("/store/autoevolve/store") / best_row[2] / "ruler.py"
        if source.is_file():
            text = source.read_text(encoding="utf-8")
            if "EVOLVE-BLOCK-START" in text:
                code = text.split("EVOLVE-BLOCK-START")[1].split("EVOLVE-BLOCK-END")[0]
    conn.close()
    result = {
        "run_id": run_id,
        "status": status,
        "metric": metric,
        "programs": count,
        "gate_passed_programs": passed,
        "operator_mix": operator_mix,
        "bandit": arms,
        "plateau_events": plateaus,
        "best_value": best_row[0] if best_row else None,
        "best_program": best_row[1] if best_row else None,
    }
    print(_json.dumps(result, indent=2), flush=True)
    if code:
        print("=== winning mutable region ===", flush=True)
        print(code[:2000], flush=True)
    return result
