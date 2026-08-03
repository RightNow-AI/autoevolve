# Nguyen-5 symbolic regression evaluator

## Goal

The candidate predicts the Nguyen-5 benchmark target
`sin(x^2) * cos(x) - 1` for inputs on `[-1, 1]`.

## Metrics and gate

The `finite` gate requires a finite Python float for every train and held-out row.
`r2_heldout` is the held-out coefficient of determination. `complexity` is the AST
node count in `predict`. `fitness` is `r2_heldout - 0.001 * complexity`, and higher
is better.

## Hardware and fixtures

This evaluator needs only a CPU and the Python standard library. The fixture follows
the Nguyen symbolic regression suite's Nguyen-5 formula. Seed `161803` samples 60
train inputs and 40 held-out inputs uniformly on `[-1, 1]`. Run
`fixtures/make_fixtures.py` to regenerate byte-identical data.

## Honesty

Recovering the formula is rediscovery of a known benchmark result. It is not a new
scientific law.

