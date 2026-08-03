# Bin-packing evaluator

## Goal

The candidate assigns item indexes to bins for one-dimensional capacity-constrained
packing. `AUTOEVOLVE_CELL` selects the uniform or clustered fixture family.

## Metrics and gate

The `valid` gate requires every item index exactly once. It rejects unknown indexes,
duplicates, missing items, empty bins, and bins above capacity. `bins_used` is the
exact total bin count across the selected family, and lower is better.

## Hardware and fixtures

This evaluator needs only a CPU and the Python standard library. Fixture seed
`424242` creates the uniform family. Seed `424243` creates the clustered family.
Each family contains 10 instances with 24 items and capacity 100. Run
`fixtures/make_fixtures.py` to regenerate byte-identical data.

