# Architecture search campaign

## Goal

This campaign evolves the activation rule and initialization scale of a small
one-hidden-layer NumPy classifier. The task uses a committed synthetic two-moons-like
dataset.

## Method

Training uses 32 fixed epochs, batches of 16 examples, and a fixed learning rate.
The `trained` gate requires finite loss and a final training loss below the first
loss. The measured metric is `val_loss`, and lower is better. `params` records the
number of trained scalar parameters. The proxy budget is 15 child evaluations.

## Promotion ladder

A single improving result is a proxy candidate and is always labeled a proxy win.
Promotion requires improving proxy runs from three distinct seeds. Scaled validation
is separate and must be run explicitly.

## Honesty

This is a small proxy task. A proxy win is not evidence about a larger model or a
scaled training run. Every reported result must include its exact run id.

