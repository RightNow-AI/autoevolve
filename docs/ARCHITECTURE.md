# ARCHITECTURE.md

Normative module and interface spec. Code that disagrees with this file is
wrong until this file is amended in the same commit. CLAUDE.md section 5 is the
constitution; this file is the engineering contract that implements it.

## 1. Module map and ownership

```
autoevolve/
  core/     engine, db, store, archive, islands, sampling, bandit, loop,
            evolve_blocks, replay, events           (unit U1)
  eval/     contract loader, sandbox, runner, cascade, feasibility   (U2)
  mutate/   operators diff/rewrite/agentic/crossover, distiller, models (U3)
  synth/    english -> locked contract pipeline                      (U3)
  mcp/      MCP server, thin adapter over core.engine                (U4)
  cli/      typer app, rich TUI, dashboard, render, report           (U5)
  gh/       GitHub issue mode                                        (U6)
core/types.py is the shared seam module. Every package imports from it.
Changing types.py requires updating this doc in the same commit.
```

## 2. Storage layout

One global store. The population outlives sessions and projects.

```
$AUTOEVOLVE_HOME (default ~/.autoevolve)/
  autoevolve.db        # the single SQLite store, WAL mode
  store/<sha256>/      # content-addressed candidate code (files on disk)
  discoveries/<domain>.md   # human-readable mirror of the discoveries table
$AUTOEVOLVE_ARTIFACTS_DIR (default <cwd>/autoevolve-runs)/<run_id>/
  dashboard.html  evolution.gif  evolution.mp4  lineage_poster.svg
  lineage_poster.png  report.md  latest.png
```

Write discipline: external processes write ONLY through MCP tools served by
the autoevolve server process. In-process core code writes directly. Workers
never touch the db file. SQLite runs WAL with busy_timeout 10s; core takes a
process-level write mutex around multi-statement transactions.

## 3. Schema (SQLite, exact DDL)

```sql
CREATE TABLE runs(
  id TEXT PRIMARY KEY, goal_text TEXT NOT NULL, domain TEXT NOT NULL,
  contract_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  budget_json TEXT NOT NULL, seed INTEGER NOT NULL, evaluator_ref TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE programs(
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
  parent_id TEXT REFERENCES programs(id), operator TEXT NOT NULL,
  code_ref TEXT NOT NULL, island INTEGER NOT NULL, cell_key TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE scores(
  program_id TEXT NOT NULL REFERENCES programs(id), metric TEXT NOT NULL,
  value REAL NOT NULL, stage INTEGER NOT NULL, measured_at TEXT NOT NULL,
  PRIMARY KEY(program_id, metric, stage));
CREATE TABLE edges(
  child_id TEXT NOT NULL, parent_id TEXT NOT NULL, kind TEXT NOT NULL,
  PRIMARY KEY(child_id, parent_id, kind));   -- kind: parent | crossover | migration
CREATE TABLE islands(
  run_id TEXT NOT NULL, island_id INTEGER NOT NULL, worker_hint TEXT,
  last_migration_at TEXT, PRIMARY KEY(run_id, island_id));
CREATE TABLE operators(
  domain TEXT NOT NULL, name TEXT NOT NULL, pulls INTEGER NOT NULL DEFAULT 0,
  improvements INTEGER NOT NULL DEFAULT 0, mean_gain REAL NOT NULL DEFAULT 0,
  PRIMARY KEY(domain, name));
CREATE TABLE discoveries(
  id TEXT PRIMARY KEY, domain TEXT NOT NULL, text TEXT NOT NULL,
  source_run TEXT, source_programs TEXT, created_at TEXT NOT NULL);
CREATE TABLE events(
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, seq));
CREATE INDEX idx_programs_run ON programs(run_id);
CREATE INDEX idx_scores_program ON scores(program_id);
```

Event kinds (closed set, renderers depend on it): run_opened, contract_locked,
worker_joined, parent_sampled, child_submitted, gate_failed, archive_improved,
migration, operator_update, discovery_added, plateau_detected, target_hit,
budget_exhausted, run_closed.

IDs: runs get "r" + uuid4.hex[:10], programs "p" + uuid4.hex[:10],
discoveries "d" + uuid4.hex[:10]. Timestamps are UTC ISO 8601 strings.

## 4. Shared types (autoevolve/core/types.py, exact)

types.py in this repo is the single source of truth for these shapes. The doc
lists the load-bearing ones. All are stdlib dataclasses, JSON-serializable via
their to_json/from_json helpers.

```python
StageSpec(name: str, timeout_s: float, mem_mb: int | None = None,
          cpu_s: float | None = None)
Budget(max_evals: int | None, wall_clock_s: float | None = None,
       max_cost_usd: float | None = None)      # at least one bound REQUIRED
Descriptor(name: str, metric: str, bins: int, lo: float, hi: float)
Contract(goal: str, domain: str, metric: str, maximize: bool,
         baseline: float | None, target: float | None, gate: str,
         budget: Budget, descriptors: list[Descriptor],
         feasibility: dict | None, plateau_n: int = 150)
Program(id, run_id, parent_id, operator, code_ref, island, cell_key, created_at)
EvalOutcome(gate_passed: bool, scores: dict[str, float], stage_reached: int,
            error: str | None)
Proposal(files: dict[str, str], notes: str)    # relative path -> full content
ParentBundle(parent: Program, parent_files: dict[str, str],
             inspirations: list[tuple[Program, dict[str, float]]],
             discoveries: list[str], crossover_parent: Program | None,
             crossover_files: dict[str, str] | None)
class EvalError(Exception)  # .reason: str, raised by evaluators on gate failure
```

## 5. The engine facade (core/engine.py)

One class, `Engine(home: Path | None = None)`. The MCP server and the CLI are
thin adapters over it; neither contains evolution logic. Method surface mirrors
the MCP tool set exactly:

```python
open_run(goal_text, evaluator_ref=None, budget: Budget, workers: int = 4,
         seed: int | None = None) -> dict   # {run_id, contract}
get_contract(run_id) -> Contract
join_run(run_id, runtime: str) -> dict     # {island: int}
next_parent(run_id, island: int) -> ParentBundle
submit_child(run_id, parent_id, operator: str, files: dict[str, str],
             notes: str = "") -> dict
    # {program_id, gate_passed, scores, fitness, archive_improved,
    #  best_fitness, plateau, budget_remaining}
best(run_id, k: int = 5) -> list[dict]
lineage(program_id) -> list[dict]
discoveries(domain, query: str | None = None) -> list[dict]
run_status(run_id) -> dict
    # {status, curve: [[eval_idx, best_fitness]...], plateau, budget_remaining,
    #  islands, artifacts: {gif, poster, dashboard}}
```

open_run semantics: with evaluator_ref, load the evaluator, measure the seed
baseline 3 times at stage 0 and record the median, run ceiling() if present,
lock the contract, insert the seed program (operator "seed"). Without
evaluator_ref, call synth (U3) to generate one first. If target exceeds
ceiling, mark the run status "infeasible" and return the analysis instead of
evolving. Refuse budgets with no bound at all.

submit_child semantics, in order: verify EVOLVE-BLOCK discipline against the
parent (files containing markers may only change inside marked regions; files
without markers are fully mutable; new files allowed); store code in the
content store; run the eval cascade; gate fail records fitness 0 scores and a
gate_failed event; gate pass records scores, updates the archive cell, updates
the bandit, appends events; return the result dict. Budget and plateau are
checked on every submit and flip run status when exhausted.

## 6. Evolution mechanics (U1 internals)

- Fitness: the contract metric, negated when maximize is false, so bigger is
  always better internally. Gate-failed programs never enter the archive.
- Archive (core/archive.py): MAP-elites keyed by cell_key computed from the
  contract descriptors (bin index per descriptor, joined with ","). No
  descriptors means the single cell "0". One elite per cell per run.
- Islands (core/islands.py): island count fixed at open_run (workers arg).
  join_run assigns round-robin. Migration: every 25 submissions per island,
  sampling may draw the parent from a neighbor island's elites (ring order)
  and records a migration event and edge.
- Sampling (core/sampling.py): parent = own-island cell elite with
  probability 0.8 (rank-weighted by fitness), else global archive elite.
  Inspirations = top K (default 3) elites from distinct cells excluding the
  parent. Discovery sampling: top 5 domain discoveries by recency and naive
  keyword overlap with the goal text.
- Bandit (core/bandit.py): UCB1 per domain over operators. gain = clipped
  relative fitness improvement (child - parent) / max(|parent|, eps), clip to
  [-1, 1]. improvement = gain > 0. Persistent in the operators table across
  runs. Unpulled operators are always preferred first; ties break by name.
- Determinism (core/replay.py): every stochastic draw uses
  random.Random(f"{run_seed}:{kind}:{event_seq}"). Sampling and migration
  decisions are recorded in event payloads. replay(run_id) re-derives every
  recorded choice from the db and asserts equality. No wall-clock dependence
  in any decision path.
- Plateau: no archive_improved event within contract.plateau_n submissions
  closes the run with status "plateau".
- EVOLVE-BLOCK (core/evolve_blocks.py): markers are lines containing
  "EVOLVE-BLOCK-START" / "EVOLVE-BLOCK-END". frozen_equal(parent_text,
  child_text) -> bool compares text outside marked regions.

## 7. Eval seam (U2 public API)

```python
eval.contract.load_evaluator(evaluator_dir: Path) -> Evaluator
    # Evaluator: .dir, .stages: list[StageSpec], .gate: str,
    #            .ceiling() -> dict | None, .spec_text: str
eval.cascade.run_cascade(evaluator, candidate_dir: Path) -> EvalOutcome
eval.sandbox.run_stage(evaluator_dir, candidate_dir, stage: int,
                       spec: StageSpec) -> dict[str, float]   # raises EvalError
```

The sandbox is a subprocess (`python -m autoevolve.eval.runner`) with: temp
copy of the candidate dir as cwd, wall-clock kill at spec.timeout_s, env
scrubbed to an allowlist, network disabled by a socket block installed before
the evaluator module loads, memory and cpu rlimits on POSIX and best-effort on
Windows (documented in CONTRACT.md). Evaluator code and candidate code NEVER
run in the engine process. The runner prints one JSON object on stdout:
{"ok": true, "metrics": {...}} or {"ok": false, "reason": "..."}.

## 8. Mutate seam (U3 public API)

```python
mutate.registry.get_operator(name: str) -> Operator   # diff|rewrite|agentic|crossover
class Operator(Protocol):
    name: str
    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal
OperatorContext(contract: Contract, rng: random.Random,
                endpoint_cheap, endpoint_strong,   # ModelEndpoint | None
                evaluate_locally: Callable[[dict[str, str]], EvalOutcome],
                workdir: Path)
mutate.models.resolve_endpoint(tier: Literal["cheap", "strong"]) -> ModelEndpoint | None
    # resolution order: AUTOEVOLVE_LOCAL_BASE_URL (+AUTOEVOLVE_LOCAL_MODEL)
    # -> OPENAI_BASE_URL/OPENAI_API_KEY (+AUTOEVOLVE_MODEL_CHEAP/_STRONG)
    # ModelEndpoint.chat(messages: list[dict], max_tokens=4096,
    #                    temperature=0.7) -> str        (httpx, OpenAI-compatible)
mutate.distiller.distill_run(engine, run_id, top_k: int = 5) -> list[dict]
synth.pipeline.synthesize(goal_text: str, workdir: Path,
                          endpoint) -> Path   # returns evaluator_dir
```

diff operator: SEARCH/REPLACE blocks against parent files, cheap endpoint.
rewrite: full-file rewrite, strong endpoint. agentic: headless
`claude -p` / `codex exec` subprocess in a scratch workspace seeded with
parent, contract, failure history, discoveries; it may call
ctx.evaluate_locally while reasoning. crossover: merges bundle.parent and
bundle.crossover_parent (which sampling guarantees is from another island).
The core loop (core/loop.py) drives operators; MCP workers run the same cycle
agent-side per skill/SKILL.md.

## 9. The worker cycle (normative, mirrors CLAUDE.md 5.2)

1. next_parent -> ParentBundle
2. operator from the bandit (engine chooses and reports it in the bundle
   payload for loop workers; MCP workers may request or override with notes)
3. propose -> files
4. submit_child -> cascade result
5. repeat until run_status says closed (target_hit, budget, plateau)

Every cycle summary printed by workers ends with the artifact paths from
run_status. A failed run explains why in one paragraph in report.md.

## 10. Model access and headless agents

OpenAI-compatible chat completions only, via httpx. No provider SDKs. The
zero-config demo path expects a local endpoint at AUTOEVOLVE_LOCAL_BASE_URL
and never requires an API key. Headless agentic mutation shells out to
`claude -p` or `codex exec`; flags are verified against live docs in U4 and
recorded in mutate/agentic.py docstrings with the doc URL.

## 11. Rendering (U5, amends from CLAUDE.md section 17)

cli/render.py renders ONLY from the events + programs + scores tables, never
from session memory. Deterministic incremental layered DAG layout: node
positions depend only on insertion order and lineage, so frames grow without
reshuffling. Matplotlib + pillow only. Public API:

```python
cli.render.render_all(home: Path, run_id: str, out_dir: Path,
                      live: bool = False) -> dict   # {gif, mp4, poster_svg,
                                                    #  poster_png, dashboard}
```

## 12. Testing law

Every unit ships tests that run headless on CPU in CI. Anything needing a
model endpoint gets a fake endpoint fixture (tests/fakes.py, a deterministic
ModelEndpoint stand-in). Anything needing GPU is CPU-mocked. The full suite
must pass on Windows and Linux.
