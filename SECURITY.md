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

**Not defended, by design.** The sandbox is a containment boundary for
accidental damage and runaway resource use, not a hostile code jail. A
native extension making raw syscalls can bypass the Python level network
block. On Windows the wall clock timeout is the enforced bound because
resource limits are not available. Do not run untrusted adversarial code
and rely on this alone. For hostile input, run autoevolve inside a
container or a virtual machine.

These limits are documented in docs/CONTRACT.md section 4 rather than
hidden, because a sandbox people misjudge is worse than one they understand.

## Scope

In scope: sandbox escapes reachable from ordinary evaluator or candidate
code, the GitHub issue mode consent gate, secret leakage into the database,
events, reports, or dashboards, and anything that lets a run exceed its
declared budget.

Out of scope: resource exhaustion by code you wrote and ran yourself, and
the native syscall bypass documented above.
