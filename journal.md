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
