# Contributing to autoevolve

Thanks for wanting to build on this. The rules below exist so results stay
real, not to slow you down.

## The two laws

1. **Measured or null.** Every number in code, docs, comments, or a pull
   request body comes from a real run and carries its run id. A benchmark
   claim without a run id is a defect. See docs/HONESTY.md.
2. **The gate comes first.** Correctness is checked before any score counts.
   Evolution will happily delete the work to make a number go up, so never
   weaken a gate to make a candidate pass.

## Setup

```sh
git clone https://github.com/RightNow-AI/autoevolve && cd autoevolve
uv sync --dev
uv run pytest -q
uv run ruff check .
```

Python 3.11 or newer. Everything runs on CPU. Tests must pass on Linux and
Windows, which CI enforces on both.

## Before you open a pull request

- `uv run ruff check .` is clean.
- `uv run pytest -q` is green, including the claims lint.
- Every fix ships a test that fails against the broken code. A fix without a
  failing-first test is not finished.
- Docs that specify behavior are updated in the same commit as the behavior.
  docs/ARCHITECTURE.md, docs/CONTRACT.md, and docs/CAMPAIGNS.md are
  normative: code that disagrees with them is wrong until they are amended.
- Conventional commit subjects, for example `fix(core): ...`.

## Where things live

| area | what it owns |
|---|---|
| `autoevolve/core` | evolution state, archive, islands, bandit, replay |
| `autoevolve/eval` | evaluator loading, sandbox, cascade, feasibility |
| `autoevolve/mutate` | operators, distiller, model endpoints |
| `autoevolve/mcp` | the MCP server, a thin adapter over the engine |
| `autoevolve/cli` | commands, TUI, renderer, dashboard, report |
| `autoevolve/gh` | GitHub issue mode |
| `evaluators/` | bundled evaluator packs |
| `campaigns/` | research campaign packs |

`autoevolve/core/types.py` is the shared seam. Changing it changes every
package, so explain why in the pull request.

## Writing an evaluator

Read docs/CONTRACT.md, then `uv run autoevolve init my-evaluator`. Declare
`METRIC` and `MAXIMIZE` so the engine never has to guess what it is
optimizing. Ship fixtures with a deterministic generator and state its seed
in spec.md. GPU evaluators declare their hardware and ship a CPU mock so CI
stays green.

## Safety rules that are not negotiable

- Candidate code always runs in the sandbox subprocess. There is no
  in-process execution path, including in tests.
- Nothing derived from a public issue executes before a maintainer applies
  the `evolve:approved` label.
- Every run requires a budget bound. Unbounded runs are refused.
- Secrets never enter the database, events, dashboards, or logs.

## Reporting something broken

Open an issue with what you ran, what you expected, what happened, and the
run id if a run was involved. `uv run autoevolve report <run_id>` produces
everything needed to reproduce.
