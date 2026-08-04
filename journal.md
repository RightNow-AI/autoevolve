# journal

One honest evidence block per completed unit. What was built, the test run
output, what is NOT done. Entries are append-only.

## U0 scaffold (commit 1361fcb, 2026-08-03)

Built: uv project on Python 3.12, package skeleton for all seven subpackages,
shared seam types in autoevolve/core/types.py, normative docs
(ARCHITECTURE.md, CONTRACT.md, HONESTY.md) written before the code they
specify, CI matrix (ubuntu + windows), Apache-2.0 license, README stub with
zero numbers.

Evidence: `uv run ruff check .` all checks passed. `uv run pytest -q`
4 passed in 0.56s.

NOT done: everything else. No engine, no eval, no operators, no surfaces.
The docs are spec, not description, until their units land.

## U1 core engine (commit 471e1e7, 2026-08-03)

Built: db (WAL, exact normative DDL), content store, MAP-elites archive,
islands with ring migration, sampling, persistent UCB1 bandit, EVOLVE-BLOCK
enforcement, Engine facade, worker loop, deterministic replay. Eval seams
injectable so core stays stdlib-only.

Evidence: ruff clean, 60 tests pass in 3.3s in the lane; full merged suite
green on main.

NOT done: cost accounting is a raw optional score, no normative cost field.
Concurrent same-parent submissions associate FIFO from events.

## U2 eval sandbox (commit in merge 5213c39 lineage, 2026-08-03)

Built: evaluator loader with all evaluator code isolated to a runner
subprocess (describe, ceiling, evaluate modes), sandbox with temp copy, env
scrub allowlist, socket block, wall-clock tree kill on Windows and rlimits on
POSIX, cascade with gate-first semantics, direction-aware feasibility.

Evidence: ruff clean, 18 tests pass including a real timeout kill in under
15s and a real blocked network attempt on Windows.

NOT done: memory and cpu rlimits are POSIX-only, documented in CONTRACT.md.

## U3 operators and synthesis (commit lineage of merge, 2026-08-03)

Built: diff (search-replace), rewrite, agentic (verified claude -p and codex
exec command shapes), model-free deterministic crossover, discovery
distiller, model endpoint resolution, english-to-evaluator synthesis with one
validation retry. chat() adapts max_completion_tokens and temperature on 400s
naming them, verified against the live Azure v1 route.

Evidence: ruff clean, 42 tests pass including argument-exact agentic command
assertions and adaptive-param round trips.

NOT done: agentic operator not yet exercised against live claude or codex in
a run; that is U8 proof territory.

## U4 MCP server and skill (2026-08-03)

Built: nine-tool MCP server as a thin Engine adapter on the installed mcp
2.0.0 SDK, structured error dicts, stdio and streamable-http serving,
SKILL.md plus reference.md plus AGENTS.md teaching the same worker loop to
Claude Code and Codex.

Evidence: ruff clean, 12 tests pass including in-memory client round trips
for every tool.

NOT done: live transport smoke happens in U8 proof 3.

## U5 CLI and visualization (2026-08-03)

Built: seven typer commands, rich live TUI, deterministic incremental DAG
renderer (gif, optional mp4 behind ffmpeg detection, poster svg and png,
live png), self-contained light and dark dashboard.html, db-derived
report.md for all four terminal states.

Evidence: ruff clean, 14 tests pass, autoevolve --help boots the real
entrypoint.

NOT done: artifacts from a real evolution run land in U8.

## U6 GitHub issue mode (commit 35c92f5, 2026-08-03)

Built: proposal-only opened handler with no import path to execution
(asserted by a test), writer-verified evolve:approved gating, milestone
comments, terminal comment, artifact-embedded terminal PR, workflow
template.

Evidence: ruff clean, 27 tests pass, all offline against fakes.

NOT done: end-to-end run against a git fixture repo is U8 proof 1.

## U7 evaluator packs (commit 504457a, 2026-08-03)

Built: python-speedup (pure-python image pipeline, equality gate, measured
speedup), triton-kernel (parity gate, honest mock_ metrics without GPU,
roofline ceiling when GPU present), routing-heuristic (permutation gate,
exact tour cost), symbolic-regression (Nguyen-7, held-out r2, complexity
penalty).

Evidence: ruff clean, 19 tests pass; all four fixture generators re-run by
the orchestrator and produced byte-identical files. Line-ending class of bug
killed repo-wide with .gitattributes eol=lf.

NOT done: no real evolution numbers exist yet anywhere, per HONESTY.md.

## U9 campaign packs (2026-08-03)

Built: campaigns/ kernel-frontier, arch-search, algorithm-frontier,
equation-discovery per docs/CAMPAIGNS.md, the campaign runner (list, run,
report) with db-derived ladder labeling, and the claims lint enforced inside
the test suite. The lint immediately caught an unmarked illustrative
multiplier in the spec doc, which is exactly its job.

Evidence: ruff clean, 15 campaign and lint tests pass, `autoevolve campaign
list` shows all four packs.

NOT done: kernel-frontier cells share the triton evaluator fixtures because
the evaluator does not read AUTOEVOLVE_CELL yet; documented in its spec. No
campaign has been run beyond proxy smoke in tests. (Resolved later the same
day by the review fix wave: the pack now selects fixture groups per cell.)

## U8 proofs (2026-08-03)

Proof 2, the flagship demo: run r8d0a8d799d locked contract metric speedup
target 10 and closed target_hit at a measured 10.90x on evaluation 16 of a
200 budget, diff operator only, seed 47. Artifacts (gif, mp4, posters,
dashboard, report) rendered from the store and shipped in docs/gallery. The
winning program discovered numpy vectorization inside the EVOLVE-BLOCK and
wrapped arrays to keep the equality gate passing.

Proof 3, any agent joins: run rda6528a177 was served over MCP streamable
HTTP and worked simultaneously by a real Claude Code session and a real
Codex session; the islands table records both runtimes and both submitted
gate-checked programs. PROOF-3 PASS with 4 non-seed programs.

Proof 1, issue mode end to end: the action entrypoint ran against a local
fake GitHub API with everything else real: consent-gated proposal comment,
approval verification, baseline, locked contract, evolution, terminal
comment, artifact-committing PR calls. PROOF-1 PASS run r59ff8810dd: one
proposal comment with zero pre-approval execution, then a full approved
run closing budget_exhausted with a terminal comment and a four-file PR.

Live-run finds that became fixes during proofs: bytecode in candidate
loading, the CLI loop wiring, the wrong-metric contract guess (METRIC and
MAXIMIZE declarations), max_completion_tokens and temperature endpoint
adaptivity, skipped-cycle resilience and logging, serve --http port landing
in the home parameter, and the issue-mode operators allowlist. Every one
shipped with a regression test.

## Adversarial review wave (2026-08-03)

Twenty-six agents, five dimensions, one adversarial verifier per finding:
20 confirmed findings, 1 refuted, recorded in
docs/reviews/2026-08-03-adversarial-review.md. All 20 fixed the same day
across two Codex fix lanes (core mechanics, sandbox layer) and one inline
wave (consent boundary, PR gate filter, config target, metric shadowing,
lint coverage, doc drift), except one accepted known-minor (synthetic viz
fixture payloads). Suite grew from 214 to 233 tests, all green with ruff
clean on every merge.

## Live-run defect wave (2026-08-04)

Three defects found by watching real runs, none visible to a green suite,
recorded in docs/reviews/2026-08-04-live-run-defects.md. The describe probe
and the sandbox built their child environments from two private copies of one
allowlist, and the copies drifted, so a frontier pack that reads its cell at
import time as docs/FRONTIER.md requires could not be described and therefore
could not run at all. The Ramsey campaign sat at zero programs behind a store
that looked idle. The agentic operator was judging its mutation by the agent's
exit code, so seven consecutive cycles discarded finished edits because a
SessionEnd hook in the host's plugin config failed in a headless subprocess.
The Modal status report dropped any store it could not read, which is what let
the first defect hide.

All three fixed with regression tests that fail against the old code. Suite
grew from 298 to 304.

Every one of the three lived in the seam between a component and the thing
that reports on it, not in the logic any test covered.
