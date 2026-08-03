# autoevolve run report: rda6528a177

## Outcome

- Status: `open`
- Domain: `python-speedup`
- Goal: dual worker proof: speed up the image pipeline

The run is still open after 4 evaluations. The current best measured speedup is 2.28617 from program pc1166601a5.

## Locked contract

```json
{
  "baseline": 0.9990785766269155,
  "budget": {
    "max_cost_usd": null,
    "max_evals": 12,
    "wall_clock_s": null
  },
  "descriptors": [],
  "domain": "python-speedup",
  "feasibility": null,
  "gate": "correct",
  "goal": "dual worker proof: speed up the image pipeline",
  "maximize": true,
  "metric": "speedup",
  "plateau_n": 150,
  "target": null
}
```

## Measured baseline

- Program: `pd424a0f7f7`
- `speedup`: 0.999079

## Best found

- Program: `pc1166601a5`
- Primary metric `speedup`: 2.28617
- Score `candidate_ms`: 5.5349
- Score `correct`: 1

## Fitness milestones

| evaluation | program | best speedup |
|---:|---|---:|
| 0 | `pd424a0f7f7` | 0.999079 |
| 1 | `p7ff70ce7af` | 1.94581 |
| 3 | `pe471824652` | 2.05605 |
| 4 | `pc1166601a5` | 2.28617 |

## Artifacts

- Dashboard: `dashboard.html (not generated yet)`
- Evolution GIF: `evolution.gif (not generated yet)`
- Evolution MP4: `evolution.mp4 (not generated yet)`
- Lineage poster SVG: `lineage_poster.svg (not generated yet)`
- Lineage poster PNG: `lineage_poster.png (not generated yet)`
- Report: `report.md`

## Replay

Replay run `rda6528a177` with recorded seed `99`. The database contains the
ordered programs, scores, lineage edges, and append-only events needed to reconstruct this result.

## Version

`autoevolve 0.1.0`
