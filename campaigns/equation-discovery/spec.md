# Equation discovery campaign

## Goal

This campaign asks symbolic regression to recover the Nguyen-5 target
`sin(x^2) * cos(x) - 1` from committed train and held-out samples.

## Method

The `finite` gate checks every train and held-out prediction. The report records
`r2_heldout`, candidate AST `complexity`, and the combined `fitness`. The fixture
generator uses seed `161803`, with 60 train points and 40 held-out points on
`[-1, 1]`.

## Promotion ladder

One completed run is a proxy candidate. Promotion requires improving runs from three
distinct seeds. Scaled validation requires an explicit run and is never inferred.

## Honesty

Nguyen-5 is part of the published Nguyen symbolic regression benchmark suite.
Recovering it is a rediscovery of a known result. It is not a novelty claim. The
report labels a strong held-out recovery as rediscovery and includes its run id.

