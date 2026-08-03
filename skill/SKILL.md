---
name: autoevolve-worker
description: Use this skill whenever asked to join an autoevolve run, evolve code toward a measured target, work an evolution population, mutate a candidate under EVOLVE-BLOCK rules, or report measured autoevolve progress and artifacts.
---

# Autoevolve worker

## What autoevolve is

Autoevolve runs a population of code candidates against a locked evaluator contract. Workers
propose mutations, while the Engine measures every accepted child and records the population.
The run ends with a target hit or evidence for the best ceiling reached within its budget.

## The contract

Read `get_contract` before joining or editing anything. The metric, direction, correctness gate,
target, descriptors, and budget are law for the life of the run. Do not replace them with a proxy,
and do not promise an unmeasured outcome. The product promise is the target or an evidence-backed
ceiling.

Every tool can return this error dictionary:

```json
{"error": true, "kind": "ValueError", "message": "What failed and how to recover."}
```

Check `error` before using any result. For full argument and return tables, read `reference.md`.

## Connect the worker

Install this directory as the project skill at `.claude/skills/autoevolve-worker/`. Copy it or
symlink it so that `.claude/skills/autoevolve-worker/SKILL.md` exists.

Register the stdio server in Claude Code:

```text
claude mcp add --transport stdio autoevolve -- uv run autoevolve serve
```

Or start the HTTP server and register its Streamable HTTP endpoint:

```text
claude mcp add --transport http autoevolve http://127.0.0.1:8747/mcp
```

Project scope may be recorded in `.mcp.json`. The HTTP type is `http`:

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

## The worker loop

Use this concrete cycle:

1. If no run exists, call `open_run` with at least one budget bound.
2. Call `get_contract`. State the metric, direction, gate, target, and configured budget plainly.
3. Call `join_run` once for this worker and keep the returned island assignment.
4. Call `next_parent` for that island.
5. Read every parent file, inspiration, score, file excerpt, discovery, operator hint, and crossover
   field before deciding on a mutation.
6. Choose a cheap diff or an agentic dive. Change only what the evidence supports.
7. Call `submit_child` with full file contents and honest notes.
8. Read the returned gate result and measured scores. A failed gate is information. Read its reason.
9. Call `run_status`. End the cycle summary with the GIF, poster, and dashboard paths.
10. Repeat from `next_parent` until `run_status` says the run is closed.

### `open_run`

```json
{
  "call": {
    "tool": "open_run",
    "arguments": {
      "goal_text": "Reduce p95 latency without changing results",
      "evaluator_ref": "evaluators/latency",
      "max_evals": 200,
      "wall_clock_s": null,
      "max_cost_usd": null,
      "workers": 4,
      "seed": 17
    }
  },
  "result": {
    "run_id": "run_01",
    "contract": {"metric": "p95_ms", "maximize": false, "gate": "exact parity"}
  }
}
```

### `get_contract`

```json
{
  "call": {"tool": "get_contract", "arguments": {"run_id": "run_01"}},
  "result": {
    "goal": "Reduce p95 latency without changing results",
    "domain": "python-speedup",
    "metric": "p95_ms",
    "maximize": false,
    "baseline": 42.1,
    "target": 20.0,
    "gate": "exact parity",
    "budget": {"max_evals": 200, "wall_clock_s": null, "max_cost_usd": null},
    "descriptors": [],
    "feasibility": null,
    "plateau_n": 150
  }
}
```

### `join_run`

```json
{
  "call": {
    "tool": "join_run",
    "arguments": {"run_id": "run_01", "runtime": "claude-code"}
  },
  "result": {"island": 2}
}
```

### `next_parent`

```json
{
  "call": {"tool": "next_parent", "arguments": {"run_id": "run_01", "island": 2}},
  "result": {
    "parent": {"id": "prog_18", "operator": "targeted_diff", "island": 2},
    "parent_files": {
      "solution.py": "# EVOLVE-BLOCK-START\ndef solve(data):\n    return parse(data)\n# EVOLVE-BLOCK-END\n"
    },
    "inspirations": [
      {
        "program": {"id": "prog_11", "operator": "profile_guided"},
        "scores": {"p95_ms": 27.4},
        "files_excerpt": {"solution.py": "cached = lookup[key]"}
      }
    ],
    "discoveries": ["Repeated parsing dominated the last three valid children."],
    "operator_hint": "profile_guided"
  }
}
```

### `submit_child`

```json
{
  "call": {
    "tool": "submit_child",
    "arguments": {
      "run_id": "run_01",
      "parent_id": "prog_18",
      "operator": "profile_guided",
      "files": {"solution.py": "# EVOLVE-BLOCK-START\noptimized()\n# EVOLVE-BLOCK-END\n"},
      "notes": "Cache the parsed lookup after the bundle showed repeated parsing cost."
    }
  },
  "result": {
    "program_id": "prog_19",
    "gate_passed": true,
    "scores": {"p95_ms": 24.8},
    "fitness": 24.8,
    "archive_improved": true,
    "best_fitness": 24.8,
    "plateau": false,
    "budget_remaining": {"max_evals": 181}
  }
}
```

### `best`

```json
{
  "call": {"tool": "best", "arguments": {"run_id": "run_01", "k": 2}},
  "result": [
    {"program_id": "prog_19", "fitness": 24.8, "scores": {"p95_ms": 24.8}},
    {"program_id": "prog_18", "fitness": 26.1, "scores": {"p95_ms": 26.1}}
  ]
}
```

### `lineage`

```json
{
  "call": {"tool": "lineage", "arguments": {"program_id": "prog_19"}},
  "result": [
    {"program_id": "prog_seed", "parent_id": null, "operator": "seed"},
    {"program_id": "prog_18", "parent_id": "prog_seed", "operator": "targeted_diff"},
    {"program_id": "prog_19", "parent_id": "prog_18", "operator": "profile_guided"}
  ]
}
```

### `discoveries`

```json
{
  "call": {
    "tool": "discoveries",
    "arguments": {"domain": "python-speedup", "query": "parsing"}
  },
  "result": [
    {
      "id": "d1234567890",
      "domain": "python-speedup",
      "text": "Cache immutable parse results after the parity gate.",
      "source_run": "r1234567890",
      "source_programs": ["p1234567890"],
      "created_at": "2026-08-03T00:00:00+00:00"
    }
  ]
}
```

### `run_status`

```json
{
  "call": {"tool": "run_status", "arguments": {"run_id": "run_01"}},
  "result": {
    "status": "running",
    "curve": [[0, 42.1], [19, 24.8]],
    "plateau": false,
    "budget_remaining": {"max_evals": 181},
    "islands": {"active": 4},
    "artifacts": {
      "gif": ".autoevolve/runs/run_01/progress.gif",
      "poster": ".autoevolve/runs/run_01/poster.png",
      "dashboard": ".autoevolve/runs/run_01/dashboard.html"
    }
  }
}
```

## EVOLVE-BLOCK discipline

If a file contains EVOLVE-BLOCK markers, content outside the markers is frozen. Files without
markers are fully mutable, and new files are allowed, but do not use that rule to route around a
marked file's frozen content. Always submit full contents for every changed file.

Right:

```python
CONFIG = {"mode": "safe"}

# EVOLVE-BLOCK-START
def solve(data):
    return optimized_path(data)
# EVOLVE-BLOCK-END

PUBLIC_API = "v1"
```

Wrong:

```python
CONFIG = {"mode": "unsafe"}  # Changed outside the mutable region.

# EVOLVE-BLOCK-START
def solve(data):
    return optimized_path(data)
# EVOLVE-BLOCK-END

PUBLIC_API = "v2"  # Changed outside the mutable region.
```

The Engine compares the child with the parent and rejects marker violations. Do not submit a
known violation to see whether it passes.

## Choosing depth

Use the cheap diff path for a small, local, evidence-backed change. Typical cases include one
branch condition, one allocation, one cache, or one focused algorithm replacement. Keep the diff
narrow and submit quickly so measurement can guide the next cycle.

Use an agentic dive when the bundle shows a complex failure, unclear bottleneck, interacting files,
or a plateau that small edits have not moved. Profile or inspect the relevant code, read the failure
reason, compare strong and weak inspirations, reason about the operator hint, and then edit. The
deeper path still ends with one child and one measured submission.

Before every mutation, read the inspirations and discoveries. They are the population's working
memory. They guide a proposal, but only `submit_child` can establish a score for that proposal.

## Honesty rules

- Never claim a score that did not come from `submit_child`.
- Never turn a failed correctness gate into a performance claim.
- Report a failed gate and its reason as useful information.
- Never present a discovery, local profile, or intuition as the child's measured result.
- End every cycle summary with the `run_status` artifact paths for GIF, poster, and dashboard.
- Keep the run id and program id beside every benchmark claim.
- State whether the run stopped on target, budget, plateau, infeasibility, or another recorded state.

## Stopping

Stop requesting parents when `run_status.status` says the run is closed. Call `best` and report the
best measured fitness with its program id. Print the GIF, poster, and dashboard paths from
`run_status.artifacts`, even when a path is null. Explain the recorded stop reason without turning a
budget or plateau result into a target hit.
