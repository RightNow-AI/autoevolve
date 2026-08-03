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
