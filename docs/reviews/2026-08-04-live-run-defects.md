# Defects found by watching live runs, 2026-08-04

Three failures found by instrumenting what the system did rather than reading
what it should do. All three were invisible in the test suite, green in CI, and
each one silently wasted real compute. Every one of them now has a regression
test that fails against the old code.

The pattern is the same one this project keeps meeting: a component reported
success while doing nothing, and only a direct look at its output found it.

## 1. A frontier pack could not run at all, and the failure looked like idleness

The Ramsey campaign sat at zero programs for a full day. Its store existed, its
database existed, and the status report simply did not list it, which read as
"not launched yet".

Two child processes load evaluator code: the describe probe, which imports an
evaluator to learn its contract, and the sandbox, which judges a candidate.
Each built its environment from its own private copy of one allowlist, and the
copies drifted. The sandbox learned to pass `AUTOEVOLVE_` workload
configuration so a campaign could select a cell. The describe probe never did.

docs/FRONTIER.md requires a frontier pack to read its cell at import time, so a
candidate cannot choose the instance it is judged against. Describe is an
import. A pack that followed the rule could not be described, so it could not
be loaded, so it could not run:

```
EvalError: AUTOEVOLVE_CELL must be one of k3-smoke, k4-climb, k5-frontier; got None
```

The campaign CLI had papered over this by reaching into both modules at runtime
and widening their private allowlists for the duration of a run. That covered
campaign runs only. Every long search is launched with a plain `autoevolve
run`, which still lost the cell.

Fixed by giving both paths one rule in `autoevolve/eval/childenv.py`. The
campaign loader now rejects a cell variable that could never arrive, rather
than letting it select nothing while the run looks healthy.

## 2. The agentic operator did its job and the work was thrown away every time

The operator that spawns a full coding session is the one structural advantage
this project has over a single-call mutation. It had never once produced an
accepted child.

It was not failing to edit. Claude Code read the prompt, made a genuine
optimization to the mutable region, and exited 1 because a `SessionEnd` hook in
the host's plugin config could not resolve `CLAUDE_PLUGIN_ROOT` in a headless
subprocess:

```
claude agent exited 1: SessionEnd hook [node "${CLAUDE_PLUGIN_ROOT}/hooks/session-end-cleanup.mjs"] failed: Hook cancelled
```

The operator judged the mutation by that exit code. Seven consecutive cycles
recorded `skipped`, the run never left its seed, and the discarded edits were
real work: zip-based sliding windows in place of index arithmetic, a walrus
operator to reuse the Sobel gradient terms, a locally bound `append`.

Fixed twice over. The claude command now disables session hooks, because a
mutation subprocess has no business running a human's interactive workflow
hooks. The `--bare` flag also skips them but forces API key auth and breaks an
OAuth host, so it is not usable here. More durably, a nonzero exit no longer
discards the work: if files changed that is a proposal and the gate decides,
with the exit code recorded in the notes; if nothing changed, the exit code and
stderr are reported as the failure they are.

## 3. The status report hid the stores it could not read

A store with no database, or one whose run table read as empty, was dropped
from the report entirely. A launched-and-broken run was therefore
indistinguishable from one never launched, which is what let defect 1 hide for
a day.

The read also used a read-only connection. A read-only connection cannot build
the shared-memory index a write-ahead log needs, so a live run can read back as
empty. The database and its log are now copied somewhere writable before
reading, and every skipped store reports its reason.

## Standing lesson

Three for three, the bug was not in the logic under test. It was in the seam
between a component and the thing that reports on it: an environment allowlist
that existed twice, an exit code standing in for a diff, and a status report
that treated unreadable as absent. Tests covered the logic. Nothing covered the
seams until something was watched end to end.
