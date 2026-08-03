# Autoevolve MCP reference

Every tool is a thin adapter over one Engine method. Tool results contain JSON-safe values. Check
for the error dictionary before reading the happy-path shape.

## Error convention

| field | type | meaning |
|---|---|---|
| `error` | `true` | Marks the result as an error. |
| `kind` | `str` | Python exception type raised by the Engine or result shaper. |
| `message` | `str` | User-facing failure and recovery text. |

Example:

```json
{
  "error": true,
  "kind": "ValueError",
  "message": "Unknown run id. List existing runs with the autoevolve CLI, or call open_run to create a run."
}
```

## `open_run`

| argument | type | default | meaning |
|---|---|---|---|
| `goal_text` | `str` | required | Measured optimization goal in plain language. |
| `evaluator_ref` | `str \| null` | `null` | Evaluator reference. The Engine synthesizes one when absent. |
| `max_evals` | `int \| null` | `null` | Maximum submitted evaluations. |
| `wall_clock_s` | `float \| null` | `null` | Maximum wall-clock seconds. |
| `max_cost_usd` | `float \| null` | `null` | Maximum measured model or compute cost. |
| `workers` | `int` | `4` | Planned worker count. |
| `seed` | `int \| null` | `null` | Replay seed. |

At least one budget bound is required. The adapter constructs
`Budget(max_evals, wall_clock_s, max_cost_usd)` and calls `Engine.open_run` once.

| return | type | meaning |
|---|---|---|
| `run_id` | `str` | New run identifier. |
| `contract` | `dict` | Locked contract. |
| other fields | JSON-safe | Engine feasibility or opening details, when present. |

## `get_contract`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Run to inspect. |

Returns the serialized `Contract` dictionary:

| field | type |
|---|---|
| `goal` | `str` |
| `domain` | `str` |
| `metric` | `str` |
| `maximize` | `bool` |
| `baseline` | `float \| null` |
| `target` | `float \| null` |
| `gate` | `str` |
| `budget` | `{max_evals, wall_clock_s, max_cost_usd}` |
| `descriptors` | `list[dict]` |
| `feasibility` | `dict \| null` |
| `plateau_n` | `int` |

## `join_run`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Run to join. |
| `runtime` | `str` | required | Worker runtime label. |

| return | type | meaning |
|---|---|---|
| `island` | `int` | Island assigned to this worker. |

## `next_parent`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Active run. |
| `island` | `int` | required | Island returned by `join_run`. |

| return field | type | meaning |
|---|---|---|
| `parent` | `dict` | All `Program` fields. |
| `parent_files` | `dict[str, str]` | Full parent file contents. |
| `inspirations` | `list[dict]` | Program, scores, and file excerpts for each inspiration. |
| `discoveries` | `list[str]` | Relevant population findings. |
| `operator_hint` | `str \| null` | Engine-selected operator hint. |
| `crossover_parent` | `dict`, optional | Crossover `Program` fields. |
| `crossover_files` | `dict[str, str]`, optional | Full crossover parent files. |

Each inspiration is:

```json
{
  "program": {
    "id": "prog_11",
    "run_id": "run_01",
    "parent_id": "prog_03",
    "operator": "targeted_diff",
    "code_ref": "sha256:7f6c48a72f5f2c9a",
    "island": 1,
    "cell_key": "size=2",
    "created_at": "2026-08-03T12:00:00Z"
  },
  "scores": {"metric_name": 1.25},
  "files_excerpt": {"relative/path.py": "relevant excerpt"}
}
```

`files_excerpt` is `{}` when the Engine's shared `ParentBundle` supplies only its documented
`(Program, scores)` pair.

## `submit_child`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Active run. |
| `parent_id` | `str` | required | Parent program selected by `next_parent`. |
| `operator` | `str` | required | Operator used for this mutation. |
| `files` | `dict[str, str]` | required | Full contents of every changed or new relative path. |
| `notes` | `str` | `""` | Concise mutation reasoning. |

A submission that violates EVOLVE-BLOCK discipline is rejected without
creating a program: the result is `{"rejected": true, "reason": str}`. Check
for the `rejected` key before reading score fields.

| return field | type |
|---|---|
| `program_id` | `str` |
| `gate_passed` | `bool` |
| `scores` | `dict[str, float]` |
| `fitness` | `float` |
| `archive_improved` | `bool` |
| `best_fitness` | `float` |
| `plateau` | `bool` |
| `budget_remaining` | `dict` |

Files that contain EVOLVE-BLOCK markers may change only inside their marked regions. Files without
markers are fully mutable. New files are allowed.

## `best`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Run to rank. |
| `k` | `int` | `5` | Maximum records to return. |

Returns `list[dict]` in Engine best-first order. Each record is passed through after JSON-safe
shaping. An Engine exception returns the standard error dictionary instead of a list.

## `lineage`

| argument | type | default | meaning |
|---|---|---|---|
| `program_id` | `str` | required | Program whose ancestry is requested. |

Returns `list[dict]` of Engine lineage nodes. An Engine exception returns the standard error
dictionary instead of a list.

## `discoveries`

| argument | type | default | meaning |
|---|---|---|---|
| `domain` | `str` | required | Discovery domain. |
| `query` | `str \| null` | `null` | Optional text filter. |

Returns `list[dict]` of discovery records. An Engine exception returns the standard error
dictionary instead of a list.

## `run_status`

| argument | type | default | meaning |
|---|---|---|---|
| `run_id` | `str` | required | Run to inspect. |

| return field | type | meaning |
|---|---|---|
| `status` | `str` | Running or recorded closure state. |
| `curve` | `list[[int, float]]` | Evaluation index and best fitness points. |
| `plateau` | `dict` | `{"limit": int, "non_improving": int, "reached": bool}`. |
| `budget_remaining` | `dict` | Remaining configured bounds. |
| `islands` | JSON-safe | Engine island summary. |
| `artifacts.gif` | `str \| null` | Progress animation path. |
| `artifacts.poster` | `str \| null` | Poster path. |
| `artifacts.dashboard` | `str \| null` | Self-contained dashboard path. |

## Transport registration

Claude Code stdio:

```text
claude mcp add --transport stdio autoevolve -- uv run autoevolve serve
```

Claude Code HTTP:

```text
claude mcp add --transport http autoevolve http://127.0.0.1:8747/mcp
```

Project-scoped HTTP entry:

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
