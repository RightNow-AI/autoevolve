# autoevolve run report: r8d0a8d799d

## Outcome

- Status: `target_hit`
- Domain: `python-speedup`
- Goal: make the image pipeline at least 10x faster with identical outputs

The run ended because it reached the target of 10 for speedup. The best measured value was 10.8969 from program pd5e4951f17 after 16 evaluations.

## Locked contract

```json
{
  "baseline": 1.026897389490496,
  "budget": {
    "max_cost_usd": null,
    "max_evals": 200,
    "wall_clock_s": null
  },
  "descriptors": [],
  "domain": "python-speedup",
  "feasibility": null,
  "gate": "correct",
  "goal": "make the image pipeline at least 10x faster with identical outputs",
  "maximize": true,
  "metric": "speedup",
  "plateau_n": 150,
  "target": 10.0
}
```

## Measured baseline

- Program: `pf69721c702`
- `speedup`: 1.0269

## Best found

- Program: `pd5e4951f17`
- Primary metric `speedup`: 10.8969
- Score `candidate_ms`: 1.0944
- Score `correct`: 1

## Fitness milestones

| evaluation | program | best speedup |
|---:|---|---:|
| 0 | `pf69721c702` | 1.0269 |
| 1 | `p07ccb4825a` | 6.1388 |
| 4 | `pebaa19b2d8` | 6.6559 |
| 15 | `pc71f3a70f8` | 6.7253 |
| 16 | `pd5e4951f17` | 10.8969 |

## Artifacts

- Dashboard: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\dashboard.html`
- Evolution GIF: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\evolution.gif`
- Evolution MP4: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\evolution.mp4`
- Lineage poster SVG: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\lineage_poster.svg`
- Lineage poster PNG: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\lineage_poster.png`
- Report: `C:\Users\jaber\RightNow-Full\AutoEvolve\autoevolve-runs\r8d0a8d799d\report.md`

## Replay

Replay run `r8d0a8d799d` with recorded seed `47`. The database contains the
ordered programs, scores, lineage edges, and append-only events needed to reconstruct this result.

## Version

`autoevolve 0.1.0`
