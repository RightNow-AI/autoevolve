# CLAUDE.md — autoevolve

This file is the constitution of this repository. Every session, every agent, every
unit of work obeys it. When a decision here conflicts with your instinct, this file
wins. When something here is genuinely wrong, write a BLOCKERS.md entry with evidence
and stop — do not silently deviate.

---

## 1. What autoevolve is

AutoEvolve is an open, agent-native evolutionary optimization and discovery system.
A person states a goal in english — an issue, a CLI arg, a chat message. The system
synthesizes a scoring contract, measures the baseline, checks feasibility, then
evolves code toward the target with a parallel population of coding-agent workers,
and ships the result as a report or a PR.

It exists to produce new things: faster kernels, better algorithms, new model
architecture components, rediscovered and novel equations. Machines get an
optimization substrate; humans get "if you can say it, you can evolve it."

### The five invariants (the product — never trade these away)

1. **The agent IS the mutation operator.** Claude Code / Codex sessions mutate
   candidates. They can profile, read compiler output, debug the evaluator, reason
   about failures. Never reduce mutation to a blind LLM diff call as the only path.
2. **The population outlives sessions.** All evolution state lives in one store
   (SQLite) owned by the autoevolve server process. Worker sessions are stateless
   and disposable. Any MCP-speaking agent can join a run mid-flight.
3. **English in, contract out.** Before any compute burns: synthesize `evaluate.py`,
   measure the baseline, compute a feasibility ceiling where possible, and lock the
   contract. The promise is always: hit the target, OR deliver best-found plus an
   evidence-backed explanation of the ceiling. Both are successful outcomes.
4. **It gets smarter every run.** Top lineage diffs are distilled into
   natural-language discoveries persisted in the discovery ledger. Future runs
   sample relevant discoveries into mutation context. Knowledge compounds across
   runs and across problems.
5. **It evolves itself.** Operator selection is a persistent per-domain UCB bandit
   over measured child-improvement rates. The system's own strategy is under
   optimization at all times.

---

## 2. Product surfaces

| surface | who | shape |
|---|---|---|
| CLI | humans, local | `autoevolve init\|run\|watch\|join\|report\|campaign` |
| MCP server + SKILL.md | Claude Code, Codex, any MCP client | tools below; skill teaches the worker loop |
| GitHub issue mode | teams, public repos | issue → contract comment → approval label → evolution with milestone comments → PR |
| Dashboard | everyone | self-contained HTML: fitness curve, lineage tree, islands, operator stats. Written to disk, shareable |

Zero-config first run must produce a visible wow (a climbing curve) with no API key,
using a local model endpoint. Every run ends with a shareable artifact. A failed run
explains WHY (ceiling, budget, gate) in one paragraph. Silent death is a defect.

---

## 3. Repo layout

```
autoevolve/
  core/        # engine: db, archive, islands, sampling, bandit, loop
  eval/        # evaluator contract, sandbox, cascade, feasibility
  mutate/      # operators: diff, rewrite, agentic, crossover; discovery distiller
  synth/       # english -> contract pipeline
  mcp/         # MCP server (the ONLY write path to the db from outside core)
  cli/         # typer app, TUI (rich), report + dashboard generation
  gh/          # issue-mode action entrypoint + comment/PR logic
evaluators/    # bundled evaluator packs (see §7)
campaigns/     # research campaign packs (see §11)
skill/         # SKILL.md + supporting files for Claude Code
docs/          # CONTRACT.md, ARCHITECTURE.md, HONESTY.md
tests/
journal.md     # per-unit evidence log
BLOCKERS.md    # honest uncertainty log
```

---

## 4. Locked tech stack (decisions made — do not relitigate)

- **Python 3.11+, uv** for everything. Evaluators, kernels, ML proxies are Python;
  one language end-to-end. No Node, no Rust in v0.
- **typer** CLI, **rich** TUI. No heavy TUI frameworks.
- **Official `mcp` python SDK** (FastMCP server style) for the MCP server.
- **SQLite, WAL mode** as the single store. All external writes flow through the MCP
  server process (single writer). Workers NEVER touch the db file; they only call
  MCP tools. In-process core code may write directly.
- **pytest + ruff** gate every merge. CI on GitHub Actions.
- **Model access:** OpenAI-compatible endpoints from env. Resolution order:
  `AUTOEVOLVE_LOCAL_BASE_URL` (local engine, default demo path) → configured cloud
  keys. Headless agentic workers via `claude -p` and `codex exec`.
- **License Apache-2.0**, org `RightNow-AI/autoevolve`.
- Verify current CLI flags, skill format, MCP config, and `codex exec` behavior
  against LIVE docs before implementing those touchpoints. Cite the doc URL in the
  commit body. Training-data memory of these tools is presumed stale.

---

## 5. Core architecture

### 5.1 Schema (SQLite)

```
runs(id, goal_text, domain, contract_json, status, budget_json, created_at)
programs(id, run_id, parent_id, operator, code_ref, island, cell_key, created_at)
scores(program_id, metric, value, stage, measured_at)      -- measured-or-absent
edges(child_id, parent_id, kind)                            -- lineage incl. crossover
islands(run_id, island_id, worker_hint, last_migration_at)
operators(domain, name, pulls, improvements, mean_gain)     -- bandit state, persistent
discoveries(id, domain, text, source_run, source_programs, created_at)
events(run_id, seq, kind, payload_json, created_at)         -- append-only, drives UI
```

### 5.2 The loop (one worker cycle)

1. `next_parent(run)` → parent + K inspirations. Sampling: MAP-elites archive cells
   (behavior descriptors from the contract) + island-local bias; periodic migration
   between islands.
2. Operator chosen by UCB1 over `operators` for this domain.
3. Mutate (see §8). Child code lands in a content-addressed store (`code_ref`).
4. Evaluate in the sandbox cascade (§6). Correctness gate first. Gate fail ⇒ score 0.
5. `submit_child` → insert program + scores + edges, update archive cell if improved,
   update bandit, append event.
6. Repeat until: target hit, budget spent, or plateau (no archive improvement in N
   evals — N from contract, default 150).

Determinism: every run is replayable from the db (seeds, operator choices, and code
refs are all recorded). `autoevolve report` reconstructs everything from the db —
never from memory of the session.

### 5.3 Parallelism

N workers ≈ N islands. Workers are stateless loops around MCP tools, so parallelism
is: open more sessions (Claude Code, Codex, mixed) and call `join_run`. Migration
through the shared archive keeps islands cross-pollinating. There is no other
coordination mechanism; do not add locks between workers beyond the server's write
serialization.

---

## 6. Evaluator contract (docs/CONTRACT.md is the normative spec — keep in sync)

An evaluator is a self-contained folder:

```
<name>/
  spec.md          # what is measured, metrics, target semantics, hardware needs
  evaluate.py      # the contract entrypoint
  baseline/        # the seed program
  fixtures/        # correctness data (parity sets, test vectors)
```

`evaluate.py` exposes:

```python
def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]
    # raises EvalError with reason on gate failure; returns {} never
def ceiling() -> dict | None      # optional: theoretical max (e.g. roofline)
STAGES: list[StageSpec]           # cascade: cheap→expensive, each with timeout
GATE: str                         # name of the boolean correctness metric
```

Rules:
- Correctness gate BEFORE any score counts. Gate fail ⇒ score 0. No exceptions,
  no partial credit. Evolution will delete the work to make the number go up;
  the gate is what makes results real.
- Metrics are scalars, measured on this machine, this run. No inherited numbers.
- GPU evaluators declare hardware in spec.md and ship a CPU mock so CI stays green.
- Sandbox: candidates run in a subprocess with resource limits, no network, temp
  workdir, wall-clock timeout from the stage spec. Never `exec` candidate code
  in-process. Never relax this for convenience.

---

## 7. Bundled evaluator packs (v0 ships all four)

1. **python-speedup** — naive image-pipeline function; metric: speedup vs baseline;
   gate: output equality on fixtures. Demo target 10x.
2. **triton-kernel** — kernel throughput; gate: parity fixtures (all-or-nothing);
   `ceiling()` implements a roofline calc from device specs. GPU-gated, CPU-mocked.
3. **routing-heuristic** — TSP-style tour cost with exact scorer.
4. **symbolic-regression** — fit equations to a bundled public dataset; metric:
   held-out fit + complexity penalty. This is the science-rediscovery demo.

---

## 8. Mutation operators

- **(a) diff-mutate** — cheap model, SEARCH/REPLACE diffs against parent, K
  inspirations + sampled discoveries in context. Bulk throughput path.
- **(b) rewrite** — strong model, full-file rewrite. Occasional deep jumps.
- **(c) agentic-dive** — spawn a headless worker (`claude -p` / `codex exec`) with
  the parent, profiler output, failure history, and discoveries; it may run the
  evaluator locally while reasoning. Expensive, highest ceiling.
- **(d) crossover** — merge two lineages selected from different islands.

Bandit: UCB1 per domain over improvement rate; state persists in `operators` so
allocation knowledge survives across runs. Log operator stats into the dashboard.

**Discovery distiller:** at run end (and every M archive improvements), read top-K
lineage diffs, write 3–7 falsifiable, transferable statements ("tiling K to 128
beat vectorized loads on this GPU class because…") to
`~/.autoevolve/discoveries/<domain>.md` with source program ids. Sample the most
relevant entries into every operator context. This ledger is the compounding asset —
treat its quality as a first-class output.

---

## 9. Contract synthesis (english → locked contract)

Pipeline: parse goal → identify domain + metric + target + constraints → generate
`evaluate.py` + fixtures (or adopt user-provided ones) → run baseline 3x, record
median → run `ceiling()` if defined → emit the contract:

```
CONTRACT
goal: <english>
metric: <name>  baseline: <measured>  target: <value|"maximize">
gate: <correctness spec>   budget: <evals, wall-clock, $ cap>
feasibility: <ceiling + method | "unbounded — plateau detection governs">
```

If target > ceiling: STOP before evolving. Output the ceiling analysis and the
maximum plausible target. That output is a success state, not a failure.

The locked contract is immutable for the run. Changing it = new run.

---

## 10. Agent surfaces

### MCP tools (exact set for v0)

```
open_run(goal_text, evaluator_ref?, budget) -> run_id + contract
get_contract(run_id)
join_run(run_id, runtime) -> island assignment
next_parent(run_id, island) -> parent + inspirations + discoveries
submit_child(run_id, parent_id, operator, code, notes) -> scores + archive_delta
best(run_id, k)
lineage(program_id)
discoveries(domain, query?)
run_status(run_id) -> curve points, plateau state, budget remaining
```

### SKILL.md (in /skill, installed into Claude Code)

Teaches the worker loop: EVOLVE-BLOCK markers (`# EVOLVE-BLOCK-START/END` fence the
mutable region; everything else is frozen), the mutate→evaluate→submit cycle, when
to choose agentic-dive behavior, and how to read a contract. The skill must work
verbatim in Claude Code and be mirrored as an AGENTS.md section for Codex. One
harness, every agent.

---

## 11. Research campaigns — the discovery engine (this is why the project exists)

A campaign is a long-running, resumable run + a report pipeline, aimed at producing
genuinely new artifacts. Campaign packs live in /campaigns, each with spec.md,
evaluators, promotion ladder, and an honesty section. v0 ships four:

1. **kernel-frontier** — evolve Triton/CUDA kernels per op×shape×GPU cell
  (MAP-elites over the op/shape grid). Gate: parity fixtures. Score: measured
  TFLOPS or tok/s vs roofline %. Output: a public frontier table + kernels.
2. **arch-search** — evolve model architecture components (attention variants,
  MoE routing functions, norm/activation blocks) inside a fixed small training
  harness. Score: validation loss per FLOP budget on tiny proxy runs
  (seconds–minutes). Promotion ladder: proxy win → 3-seed proxy replication →
  scaled validation run → only then a claim. Proxy wins are ALWAYS labeled proxy
  wins.
3. **algorithm-frontier** — packing/scheduling/routing heuristics on public
  benchmark instances with exact scorers.
4. **equation-discovery** — symbolic regression against public scientific
  datasets; held-out validation mandatory; rediscovery of known laws is reported
  as rediscovery (it is the credibility demo, not the novelty claim).

Campaign honesty (non-negotiable):
- A "discovery" claim requires: reproducible artifact in-repo, held-out or
  replicated validation, and the exact run id. Otherwise it is "candidate".
- Never compare against a baseline you did not run in the same environment.
- Negative results get reported in the campaign log. The ledger of what failed is
  part of the compounding asset.

---

## 12. Safety and sandboxing

- All candidate execution: subprocess, resource limits (CPU, memory, wall-clock),
  no network, isolated temp workdir. Applies to evaluator-generated code paths too.
- Secrets never enter the db, events, dashboards, or logs. Redact env by default.
- Budget caps are mandatory on every run; refuse unbounded runs.
- GitHub issue mode: NEVER execute code derived from a public issue before a
  maintainer applies `evolve:approved`. The contract comment is the consent record.
- GPU campaigns run only on explicitly configured hardware, never opportunistically.

---

## 13. Honesty rules (docs/HONESTY.md mirrors this)

- Measured-or-null. Every number in README/docs/dashboards comes from
  `autoevolve report` over a real run artifact, or it does not appear.
- A benchmark claim without a run id is a defect. Fix by running or deleting.
- Failure is a first-class result: report ceiling/budget/plateau causes plainly.
- Never present a proxy-task result as an at-scale result.

---

## 14. Engineering standards

- Branch per unit, conventional commits, tests with every unit, ruff clean.
- Never `git add -A` / `git add .`; stage explicitly. Never force-push. Never rewrite
  shared history.
- <70% confidence on an external-tool decision → BLOCKERS.md entry with what you
  tried and the doc links, then move on. Guessing at integration details is a defect.
- journal.md gets one honest evidence block per completed unit: what was built, the
  test run output, what is NOT done.
- Docs first for load-bearing pieces: CONTRACT.md and ARCHITECTURE.md are written
  before the code they specify, and updated in the same commit as any change.

---

## 15. Build roadmap (work strictly in order; each unit ends merged + green)

- **U0** scaffold: uv project, layout above, CI (pytest+ruff), README stub,
  docs/CONTRACT.md + docs/ARCHITECTURE.md drafted as the spec.
- **U1** core engine: schema, archive (MAP-elites), islands + migration, sampling,
  bandit, resume, deterministic replay.
- **U2** evaluator contract + sandbox + cascade + feasibility hook.
- **U3** operators (a)–(d) + discovery distiller + model-endpoint resolution.
- **U4** MCP server (tool set in §10) + SKILL.md + AGENTS.md. Verify live docs.
- **U5** CLI + TUI + HTML dashboard + report.
- **U6** GitHub issue mode with approval gating, milestone comments, terminal PR.
- **U7** the four evaluator packs (§7), each with tests; triton pack CPU-mocked in CI.
- **U8** proofs: (1) e2e issue→PR on a fixture repo; (2) local demo ≥10x on
  python-speedup in ≤200 evals using operator (a) only; (3) one run served
  simultaneously by a Claude Code worker AND a Codex worker via `join_run`;
  (4) README gallery generated from real runs; quickstart ≤3 commands.
- **U9** campaign packs (§11) runnable end-to-end at proxy scale, with the ladder
  and honesty sections enforced in code (a claim without a run id fails CI lint).

**v0 DONE =** all U8 proofs green + full test suite green + zero unverifiable
numbers anywhere in the repo.

---

## 16. Anti-goals (do NOT build these)

- No unsandboxed execution path, ever, including "just for tests".
- No unbounded runs, no default-off budget caps.
- No LLM-as-judge evaluators in core (reward hacking); allowed only behind an
  `experimental` flag with a warning in the report.
- No generic library UX that re-implements OpenEvolve's shape. Agent-native (MCP,
  skills, issues) is the identity; the library is internal.
- No fabricated, estimated, or remembered benchmark numbers. Measured-or-null.


---

## 17. Visualization — the run must be watchable and shareable (amends U5, U8, §10)

Evolution is invisible unless rendered. Every run auto-generates three artifacts from
the `events` table (never from session memory), all reproducible via
`autoevolve render <run_id>`:

1. **evolution.gif** — auto-generated timelapse of the run. Two panels per frame:
   left = lineage graph growing node by node, right = best-score curve ticking up.
   Frames sampled at every archive improvement + every K evals (K chosen so any run
   renders to <=30s of GIF). Also emitted as .mp4. Regenerated at run end and on
   demand; `--live` writes latest.png every N seconds for screenshots/streams.
2. **lineage_poster.svg/png** — the final still: full genealogy of the winning
   program, score deltas annotated along the elite path. This is the shareable
   "money screenshot" of a run.
3. **dashboard.html** (existing U5 artifact) — same visual language, plus hover
   detail (score, operator, diff summary, island) and operator/bandit stats.

Visual language (minimal, fixed — do not decorate):
- node = program. small dot; hollow grey = failed correctness gate; filled = scored.
- color = fitness, single-hue ramp; ONE accent color reserved for the elite path
  root->best (drawn thicker). Everything else near-monochrome.
- edge = parentage; crossover edges join two parents; migration edges dashed.
- islands = vertical lanes; operator = small glyph on the node (4 glyphs, legend).
- layout = layered DAG, deterministic and INCREMENTAL: existing nodes never move
  between frames — the GIF must read as growth, not reshuffling.
- detail lives in hover (html) and the poster annotations, never as clutter on the
  animated view. Long runs render the elite spine + k-hop neighborhood, capped node
  count, never the full hairball.

Implementation: pure-python (matplotlib + pillow; own tidy-DAG layout, deterministic
seeds). No graphviz or system deps. Renderer lives in cli/render.py, built in U5.

Surfacing (this is UX, not decoration):
- every worker cycle summary (SKILL.md + AGENTS.md) ends by printing the artifact
  paths: dashboard.html, evolution.gif, poster. Claude Code and Codex workers always
  tell the human where to look.
- `run_status` (MCP, §10) gains `artifacts: {gif, poster, dashboard}` paths.
- issue mode: milestone comments attach the current GIF; the terminal PR embeds the
  poster + final GIF.
- U8 proof (5): evolution.gif + lineage_poster generated from a REAL run render
  correctly and are embedded in the README gallery.