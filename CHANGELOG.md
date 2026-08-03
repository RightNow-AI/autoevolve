# Changelog

## 0.1.0 (2026-08-03)

First public release. Every number below comes from a real run in this
repository's history and carries its run id.

### The system

- Core engine: SQLite store that outlives sessions, MAP-elites archive,
  islands with ring migration, persistent per domain UCB1 bandit over
  operators, EVOLVE-BLOCK enforcement, deterministic replay.
- Evaluator contract and sandbox: evaluator and candidate code run only in a
  subprocess with an environment allowlist, network block installed before
  any module loads, wall clock kill with process tree termination, and POSIX
  resource limits.
- Mutation operators: diff, rewrite, agentic through headless `claude -p` or
  `codex exec`, and model free crossover. Discovery distiller writes
  transferable findings that later runs sample into context.
- Contract synthesis: an english goal becomes a measured evaluator, a
  baseline, and a locked contract before any evolution compute burns.
- Agent surfaces: nine tool MCP server over stdio and streamable HTTP, a
  worker skill for Claude Code, and an AGENTS.md mirror for Codex.
- CLI: `init`, `run`, `watch`, `join`, `report`, `render`, `serve`, and
  `campaign`, with a Rich live TUI and deterministic renderers producing
  evolution.gif, an optional mp4, lineage posters, and a self contained
  dashboard.
- GitHub issue mode: an issue becomes a contract proposal comment, execution
  waits for the `evolve:approved` label from someone with write access, then
  evolution runs with milestone comments and opens a pull request with the
  artifacts embedded.
- Four evaluator packs: python-speedup, triton-kernel with an honest CPU
  mock, routing-heuristic, symbolic-regression on the Nguyen-7 benchmark.
- Four campaign packs: kernel-frontier, arch-search, algorithm-frontier,
  equation-discovery, with a promotion ladder and a claims lint that fails
  the test suite on any measured claim lacking a run id.

### Proven end to end

- Run r8d0a8d799d reached a measured 10.90x speedup at evaluation 16 of a
  200 budget on python-speedup using the diff operator alone, closing as
  target_hit against a locked target of 10.
- Run rda6528a177 was served over MCP streamable HTTP and worked
  simultaneously by a Claude Code session and a Codex session through
  `join_run`.
- Run r59ff8810dd completed the GitHub issue flow end to end: one proposal
  comment with zero pre approval execution, then an approved run with a
  terminal comment and a pull request carrying four files.

### Known limits

- The triton pack reports `mock_` metrics without a CUDA device. Real mode
  needs a CUDA GPU and torch, not Triton specifically, so any backend works.
  No throughput claim appears anywhere in this repository, because no such
  number has yet come from a run with a run id.
- Campaigns have been exercised at proxy scale only.
- The local model endpoint path is implemented and tested offline but has
  not been proven against a live local engine.
- The sandbox contains accidents, not hostile code. See SECURITY.md.
