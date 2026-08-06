# VRPTW campaign log

Append-only. Gate failures, plateaus, budget stops, and valid candidates use the
same run block format. No published solution value or route belongs in this file.

For every accepted run, record both objective components in this order:

```text
run_id: <run id>
cell: <cell>
vehicle_count: <gate-recomputed count>
total_distance: <gate-recomputed distance under DISTANCE_CONVENTION>
stop_reason: <target, budget, plateau, infeasibility, or other Engine state>
artifacts: <gif, poster, dashboard>
```

The campaign winner is selected lexicographically. Lower `vehicle_count` wins
first. `total_distance` breaks ties only when vehicle counts are equal.
