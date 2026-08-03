# Toy sorting evaluator

This evaluator measures two scalar metrics:

- `correct`: unitless correctness gate. It is `1.0` only when every selected fixture passes.
- `score`: deterministic source compactness, measured as `1000 / source characters`.

The gate is `correct`. The score is maximized, with no fixed target or theoretical ceiling.
The evaluator requires only a local Python 3.11 interpreter and no accelerator hardware.

The ten fixtures are hand-written integer-list sorting cases covering unordered input, already
sorted input, reverse order, duplicates, negative values, singleton input, and empty input.
Stage 0 uses the first three cases. Stage 1 uses all ten cases.
