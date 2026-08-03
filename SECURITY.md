# Security policy

## Reporting a vulnerability

Report privately through GitHub Security Advisories on this repository:
https://github.com/RightNow-AI/autoevolve/security/advisories/new

Do not open a public issue for a vulnerability. Expect an acknowledgement
within a few days.

## What this project's threat model actually covers

autoevolve executes generated code by design, so the boundaries matter more
than usual. Be precise about what is and is not defended.

**Defended.** Candidate and evaluator code runs only in a subprocess with a
fresh temporary working directory, a wall clock kill, an environment
scrubbed to an allowlist so secrets cannot reach it, and Python level
networking blocked before any evaluator module loads. Memory and CPU limits
apply on POSIX. Every run requires a budget bound. Nothing derived from a
public GitHub issue executes before a maintainer applies the
`evolve:approved` label, and the label applier must have write access.

**Verdict integrity.** The result the engine records is decided by the
evaluator and is unreachable from candidate code. File descriptor 1 carries
the verdict, and it is the null device for the whole time evaluator and
candidate code runs, so nothing they write can appear on it, including
writes through `sys.__stdout__` or a raw `os.write`. The descriptor is
restored only after the payload is decided, written with a raw write that
ignores tampering with `sys.stdout`, and the process leaves through
`os._exit` so no `atexit` handler can append a second verdict. The parent
reads the first verdict line and rejects a nonzero exit code. This is the
property every measured number in this project depends on, so it carries
regression tests for both forging after and forging before the real verdict.
Reports of a way to influence a recorded verdict are the highest severity
issue this project can receive.

The environment allowlist additionally passes `AUTOEVOLVE_` prefixed
workload configuration, because campaign cells select their workload through
it. `AUTOEVOLVE_HOME` is excluded, since the run database path would let a
candidate edit its own scores, as are endpoint, model, and credential shaped
names.

**Not defended, by design.** The sandbox is a containment boundary for
accidental damage and runaway resource use, not a hostile code jail. A
native extension making raw syscalls can bypass the Python level network
block. On Windows the wall clock timeout is the enforced bound because
resource limits are not available. Do not run untrusted adversarial code
and rely on this alone. For hostile input, run autoevolve inside a
container or a virtual machine.

Candidate code also shares an interpreter with the evaluator that judges it.
The verdict channel is protected, but an evaluator that trusts an object a
candidate hands back can still be misled about its own computation, for
example by a container subclass whose `__getitem__` is candidate code and
answers differently on each read. Evaluator authors must recompute every
quantity that reaches a metric and normalize candidate output into immutable
primitives in a single pass before checking anything. docs/CONTRACT.md
section 4a and docs/FRONTIER.md section 5 state the rules.

These limits are documented in docs/CONTRACT.md section 4 rather than
hidden, because a sandbox people misjudge is worse than one they understand.

## Scope

In scope: sandbox escapes reachable from ordinary evaluator or candidate
code, the GitHub issue mode consent gate, secret leakage into the database,
events, reports, or dashboards, and anything that lets a run exceed its
declared budget.

Out of scope: resource exhaustion by code you wrote and ran yourself, and
the native syscall bypass documented above.
