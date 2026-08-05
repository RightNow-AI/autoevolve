"""Give one long-lived agent control of a research campaign on Modal.

This is deliberately not the evolutionary loop. There, the loop is in charge:
it samples a parent, picks an operator, asks for one mutation, scores it, and
throws the session away. The agent never sees more than a single step and never
decides what to try next.

The evidence in this repository says that is the wrong way round. The only
genuine discovery it has produced came from an agent that wrote a parallel
search over the factorisations of 42 and ran it, unprompted. Its worst results
came from single-call operators reciting published constants. So here the agent
is in charge: it reads the problem, reads everything previously tried, decides
a strategy, writes and runs its own code for hours, and submits candidates when
it has something.

It cannot mark its own work. `submit` runs the pack's real evaluator in the
sandbox and returns the gate's verdict, so a candidate that does not verify
scores nothing no matter what the agent believes about it. The journal it keeps
is what makes successive sessions compound rather than restart.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"


def _head_sha() -> str:
    """Pin the image to the local commit, tolerating the container layout."""

    try:
        repo_root = Path(__file__).resolve().parents[1]
    except IndexError:
        return "main"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return result.stdout.strip() or "main"


COMMIT = _head_sha()

app = modal.App("autoevolve-researcher")
store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "bubblewrap")
    .pip_install("uv")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout {COMMIT}",
        f"cd {REPO_ROOT} && uv sync --frozen",
    )
)

# Verified by testing the matrix in a container: codex ignores a top level
# openai_base_url and goes to api.openai.com, which answers 401 for this key.
# wire_api = "chat" is rejected outright by 0.146. Only an explicit provider
# block using the responses transport actually completes an edit.
CODEX_CONFIG = """\
model_provider = "autoevolve"
model = "{model}"
approval_policy = "never"

[model_providers.autoevolve]
name = "autoevolve"
base_url = "{base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
"""

SUBMIT_TOOL = '''\
#!/usr/bin/env python3
"""Score a candidate directory with the pack's real evaluator.

The agent cannot mark its own work. This runs the same sandboxed cascade the
engine uses, so the verdict is the gate's and not the agent's. Every call is
appended to the journal, including the failures, because a record of what did
not work is what stops the next session repeating it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("REPO_ROOT_PLACEHOLDER")
EVALUATOR = Path("EVALUATOR_PLACEHOLDER")
JOURNAL = Path("JOURNAL_PLACEHOLDER")
BEST_FILE = Path("BEST_PLACEHOLDER")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: submit <candidate_dir> [note]")
        return 2
    candidate = Path(sys.argv[1]).resolve()
    note = " ".join(sys.argv[2:]) or "(no note)"
    if not candidate.is_dir():
        print(f"no such directory: {candidate}")
        return 2

    script = (
        "import json,sys;"
        "from pathlib import Path;"
        "from autoevolve.eval.contract import load_evaluator;"
        "from autoevolve.eval.cascade import run_cascade;"
        "ev=load_evaluator(Path(sys.argv[1]));"
        "out=run_cascade(ev, Path(sys.argv[2]));"
        "print(json.dumps({'gate_passed':out.gate_passed,'scores':out.scores,"
        "'stage_reached':out.stage_reached,'error':out.error}))"
    )
    completed = subprocess.run(
        ["uv", "run", "python", "-c", script, str(EVALUATOR), str(candidate)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    line = (completed.stdout or "").strip().splitlines()
    verdict = {"gate_passed": False, "error": (completed.stderr or "")[-600:]}
    if line:
        try:
            verdict = json.loads(line[-1])
        except json.JSONDecodeError:
            pass

    print(json.dumps(verdict, indent=2))
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\\n## {datetime.now(UTC).isoformat()}\\n"
            f"note: {note}\\n"
            f"verdict: {json.dumps(verdict)}\\n"
        )

    if verdict.get("gate_passed"):
        previous = {}
        if BEST_FILE.is_file():
            try:
                previous = json.loads(BEST_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
        metric = "METRIC_PLACEHOLDER"
        maximize = MAXIMIZE_PLACEHOLDER
        current = verdict.get("scores", {}).get(metric)
        prior = previous.get("scores", {}).get(metric)
        better = (
            current is not None
            and (prior is None or (current > prior if maximize else current < prior))
        )
        if better:
            payload = {"scores": verdict.get("scores", {}), "note": note}
            BEST_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            for path in candidate.rglob("*"):
                if path.is_file():
                    target = BEST_FILE.parent / "best_candidate" / path.relative_to(candidate)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            print(f"NEW BEST recorded: {metric}={current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@app.function(
    image=image,
    volumes={"/store": store},
    secrets=[modal.Secret.from_name("autoevolve-model")],
    timeout=60 * 60 * 12,
    cpu=8.0,
    memory=32768,
)
def research(
    evaluator: str,
    mission: str,
    cell: str = "",
    store_name: str = "research",
    hours: float = 4.0,
    rounds: int = 8,
) -> dict:
    """Run the campaign. GPU packs must reach this through `with_options`.

    A GPU pack silently falls back to its CPU mock when no device is attached,
    and a mock reports a speedup of zero rather than failing, so a campaign can
    look busy while measuring nothing. The local entrypoint attaches a GPU
    whenever one is asked for.
    """
    import json
    import os
    import shutil
    import time
    from pathlib import Path

    root = Path(f"/store/{store_name}")
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "JOURNAL.md"
    best_file = root / "best.json"
    workspace = root / "workspace"
    evaluator_dir = Path(REPO_ROOT) / evaluator
    baseline = evaluator_dir / "baseline"

    if not journal.exists():
        journal.write_text(
            f"# Research journal\n\nMission: {mission}\n", encoding="utf-8"
        )
    if not workspace.exists():
        workspace.mkdir(parents=True)
        if baseline.is_dir():
            shutil.copytree(baseline, workspace, dirs_exist_ok=True)

    # Read the pack's declared metric so the submit tool knows which direction
    # counts as better without the agent being able to redefine it.
    describe = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import json,sys;from pathlib import Path;"
            "from autoevolve.eval.contract import load_evaluator;"
            "e=load_evaluator(Path(sys.argv[1]));"
            "print(json.dumps({'metric':e.metric,'maximize':e.maximize,'gate':e.gate}))",
            str(evaluator_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **({"AUTOEVOLVE_CELL": cell} if cell else {})},
    )
    contract = {"metric": "score", "maximize": True, "gate": "correct"}
    if describe.stdout.strip():
        try:
            contract = json.loads(describe.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass

    submit_path = root / "submit"
    submit_path.write_text(
        SUBMIT_TOOL.replace("REPO_ROOT_PLACEHOLDER", REPO_ROOT)
        .replace("EVALUATOR_PLACEHOLDER", str(evaluator_dir))
        .replace("JOURNAL_PLACEHOLDER", str(journal))
        .replace("BEST_PLACEHOLDER", str(best_file))
        .replace("METRIC_PLACEHOLDER", str(contract["metric"]))
        .replace("MAXIMIZE_PLACEHOLDER", "True" if contract["maximize"] else "False"),
        encoding="utf-8",
    )
    submit_path.chmod(0o755)

    home = Path(os.path.expanduser("~"))
    (home / ".codex").mkdir(parents=True, exist_ok=True)
    (home / ".codex" / "config.toml").write_text(
        CODEX_CONFIG.format(
            model=os.environ.get("AUTOEVOLVE_MODEL_STRONG")
            or os.environ.get("AUTOEVOLVE_MODEL", "gpt-5.6-sol"),
            base_url=os.environ.get("OPENAI_BASE_URL", "").strip(),
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["CODEX_API_KEY"] = env.get("OPENAI_API_KEY", "")
    if cell:
        env["AUTOEVOLVE_CELL"] = cell

    seconds_per_round = max(600.0, (hours * 3600.0) / max(rounds, 1))
    history: list[dict] = []

    for index in range(rounds):
        best_text = best_file.read_text(encoding="utf-8") if best_file.is_file() else "none yet"
        journal_tail = journal.read_text(encoding="utf-8")[-6000:]
        brief = _brief(
            mission=mission,
            contract=contract,
            workspace=workspace,
            submit_path=submit_path,
            journal=journal,
            best_text=best_text,
            journal_tail=journal_tail,
            seconds=seconds_per_round,
            round_index=index,
            rounds=rounds,
        )
        (root / "BRIEF.md").write_text(brief, encoding="utf-8")

        started = time.time()
        try:
            completed = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "-C",
                    str(workspace),
                    brief,
                ],
                capture_output=True,
                text=True,
                timeout=seconds_per_round,
                check=False,
                env=env,
            )
            code, tail = completed.returncode, (completed.stdout or "")[-1500:]
        except subprocess.TimeoutExpired:
            code, tail = -1, "round hit its wall clock"
        history.append(
            {
                "round": index,
                "exit_code": code,
                "elapsed_s": round(time.time() - started, 1),
                "tail": tail[-500:],
            }
        )
        store.commit()

    best = json.loads(best_file.read_text(encoding="utf-8")) if best_file.is_file() else None
    store.commit()
    mocked = bool(best and best.get("scores", {}).get("mock_mode"))
    return {
        "store": store_name,
        "metric": contract["metric"],
        "maximize": contract["maximize"],
        "best": best,
        "mock_mode": mocked,
        "warning": (
            "this campaign ran without a GPU, so every measurement is a CPU mock "
            "and the metric is meaningless"
            if mocked
            else ""
        ),
        "rounds": history,
    }


def _brief(
    *,
    mission: str,
    contract: dict,
    workspace: Path,
    submit_path: Path,
    journal: Path,
    best_text: str,
    journal_tail: str,
    seconds: float,
    round_index: int,
    rounds: int,
) -> str:
    """Compose the standing brief the agent receives every round."""

    direction = "as high as possible" if contract["maximize"] else "as low as possible"
    return f"""You are running a research campaign. You are in charge of it.

MISSION
{mission}

You are round {round_index + 1} of {rounds}. You have about {seconds:.0f} seconds.
Earlier rounds wrote what they learned into the journal below, and the next
round will read what you write. Treat this as continuing one long piece of work.

WHAT COUNTS
The metric is {contract["metric"]}, and you want it {direction}. The gate is
{contract["gate"]}. Nothing counts until it passes that gate.

HOW TO SUBMIT
Run: {submit_path} <directory> "<what you tried and why>"
It runs the real evaluator in a sandbox and prints the verdict. You cannot mark
your own work, so do not reason about whether a candidate is good; submit it
and read what comes back. Every submission is journalled, including failures.

YOUR WORKSPACE
{workspace} holds the current candidate. Edit it freely. You may also write and
run whatever code you like anywhere else on this machine: search scripts,
brute-force checks, profiling, parallel processes. You have 8 cores. Actually
running things is the point, and it is the only way anyone here has found a
real result.

WHAT HAS ALREADY HAPPENED
Best so far: {best_text}

Journal, most recent last:
{journal_tail}

HOW TO SPEND THIS ROUND
Decide a strategy first and say what it is. Prefer a search you write and run
over a construction you remember: this project has four separate cases of a
model returning a published constant that passed every gate and proved nothing,
so a recalled answer is worth less than a worse one you actually found. If a
line of attack is exhausted, say so plainly in the journal and try a different
one. Append what you learned to {journal} before you finish, in a few sentences,
including anything that did not work.
"""


@app.local_entrypoint()
def main(
    evaluator: str,
    mission: str,
    cell: str = "",
    store_name: str = "research",
    hours: float = 4.0,
    rounds: int = 8,
    gpu: str = "",
) -> None:
    import json

    remote = research.with_options(gpu=gpu) if gpu else research
    result = remote.remote(
        evaluator=evaluator,
        mission=mission,
        cell=cell,
        store_name=store_name,
        hours=hours,
        rounds=rounds,
    )
    print(json.dumps(result, indent=2)[:4000])
