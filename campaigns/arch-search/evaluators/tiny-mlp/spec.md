# Tiny MLP evaluator

## Goal

The candidate supplies the activation rule and initialization scale for a small
binary classifier with one hidden layer.

## Metrics and gate

The `trained` gate requires finite loss and a lower final training loss than the
first loss. `val_loss` is binary cross-entropy on the committed validation split,
and lower is better. `params` is the number of trained scalar parameters.

Training always uses 32 epochs, batches of 16 examples, learning rate `0.08`, and
training seed `2718`.

## Hardware and fixtures

This evaluator needs a CPU and NumPy. Fixture seed `314159` creates a noisy
two-moons-like dataset. The committed split contains 96 training rows and 48
validation rows. Run `fixtures/make_fixtures.py` to regenerate byte-identical data.

