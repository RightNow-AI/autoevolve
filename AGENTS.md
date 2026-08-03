# Autoevolve agent instructions

## Constitution and scope

`CLAUDE.md` is the repository constitution. Read its relevant sections before changing product
behavior. Treat `docs/ARCHITECTURE.md` as the normative module and seam map. Keep each unit inside
its owned paths and do not move evolution logic into adapters.

## What autoevolve does

Autoevolve measures a population of code candidates against one locked contract. Workers propose
mutations, and the Engine owns validation, evaluation, ranking, storage, budgets, and closure. A run
delivers either its measured target or evidence for the best ceiling reached within its budget.

## Connect

For Codex, register autoevolve in the active MCP configuration with either:

- stdio command `uv run autoevolve serve`
- Streamable HTTP endpoint `http://127.0.0.1:8747/mcp`

Use the MCP configuration format supported by the active Codex host. Once unit U5 lands, a human or
agent may also drive the CLI surface through `autoevolve join`. Do not assume the CLI command exists
before that unit is present.

Python can connect directly to the server without a subprocess:

```python
import asyncio

from mcp import Client

from autoevolve.mcp.server import build_server


async def read_contract(run_id: str) -> object:
    async with Client(build_server()) as client:
        return await client.call_tool("get_contract", {"run_id": run_id})


asyncio.run(read_contract("run_01"))
```

Claude Code uses these client-specific registration commands:

```text
claude mcp add --transport stdio autoevolve -- uv run autoevolve serve
claude mcp add --transport http autoevolve http://127.0.0.1:8747/mcp
```

A project-scoped Claude Code HTTP entry uses `http`:

```json
{
  "mcpServers": {
    "autoevolve": {
      "type": "http",
      "url": "http://127.0.0.1:8747/mcp"
    }
  }
}
```

## Worker loop

Follow this cycle exactly:

1. Call `open_run` only when no suitable run exists. Supply at least one budget bound.
2. Call `get_contract` before joining or editing. Read the metric, direction, gate, target,
   descriptors, and budget.
3. Call `join_run` once with a useful runtime label. Keep the returned island assignment.
4. Call `next_parent` at the start of each cycle.
5. Read the full parent, every inspiration and score, every file excerpt, every discovery, the
   operator hint, and any crossover fields.
6. Choose a narrow cheap diff or a deeper evidence-gathering pass.
7. Submit one child with `submit_child`. Send full changed file contents and honest notes.
8. Read the gate result and measured scores. Do not infer a score from a local profile.
9. Call `run_status` after each submission. End the cycle summary with all artifact paths.
10. Repeat until `run_status.status` records closure.

Check every tool result for this shape before reading its normal fields:

```json
{"error": true, "kind": "ValueError", "message": "What failed and how to recover."}
```

An unknown run id error tells the caller to list existing runs through the autoevolve CLI or call
`open_run` to create one.

## Tool roles

| tool | role in the loop |
|---|---|
| `open_run` | Lock a measured contract and bounded run. |
| `get_contract` | Read the law for the run. |
| `join_run` | Receive a worker island. |
| `next_parent` | Receive one parent bundle with evidence. |
| `submit_child` | Validate, evaluate, and record one mutation. |
| `best` | Read the measured leaders. |
| `lineage` | Read recorded ancestry. |
| `discoveries` | Search reusable measured findings. |
| `run_status` | Read progress, closure, budget, and artifacts. |

## EVOLVE-BLOCK law

When a file contains EVOLVE-BLOCK markers, only content inside those markers may change. Content
outside the markers is frozen. Files without markers are fully mutable, and new files are allowed,
but never use a new file to bypass a frozen contract or public surface.

Allowed:

```python
PUBLIC_MODE = "stable"

# EVOLVE-BLOCK-START
def solve(data):
    return optimized(data)
# EVOLVE-BLOCK-END
```

Rejected:

```python
PUBLIC_MODE = "experimental"  # Frozen content changed.

# EVOLVE-BLOCK-START
def solve(data):
    return optimized(data)
# EVOLVE-BLOCK-END
```

The Engine checks the child against the selected parent. Do not submit a known marker violation.

## Choosing depth

Use a cheap diff when the evidence points to one small local change. Examples include removing one
allocation, caching one immutable result, tightening one branch, or replacing one focused loop.
Submit quickly and let measurement choose the next step.

Use an agentic dive when the failure is unclear, the bottleneck spans files, strong inspirations
disagree, or the archive has plateaued. Read the failure details, profile or inspect the relevant
path, compare inspirations, use discoveries, and reason about the operator hint before editing.
The deeper pass still produces one child for one measured submission.

Read inspirations and discoveries before every mutation. They guide the proposal. They do not prove
the current child's score.

## Honesty rules

- Never claim a score that was not returned by `submit_child`.
- Never claim a gate-failed child as a performance improvement.
- Treat a failed gate and its reason as useful evidence.
- Keep the run id and program id beside benchmark claims.
- Do not present proxy-task evidence as the locked contract's result.
- End every cycle summary with GIF, poster, and dashboard paths from `run_status.artifacts`.
- State the recorded stop reason: target, budget, plateau, infeasibility, or another Engine state.

## Stop and report

Stop requesting parents when `run_status.status` says the run is closed. Call `best`. Report the
best measured fitness with its program id, the recorded stop reason, and the GIF, poster, and
dashboard paths. A null artifact path must remain null. Do not turn budget exhaustion or plateau
into a target claim.

## Maintainers

- The Engine facade and exact shared types live in `docs/ARCHITECTURE.md` sections 4 and 5.
- The normative worker cycle lives in `docs/ARCHITECTURE.md` section 9.
- `autoevolve/mcp/server.py` stays a thin adapter. It contains no evolution logic and no database
  access.
- Tool docstrings and `skill/SKILL.md` are product UX. Keep schemas, examples, error handling, and
  behavioral guidance in sync.
- `CLAUDE.md` remains the constitution when any secondary document conflicts with it.
