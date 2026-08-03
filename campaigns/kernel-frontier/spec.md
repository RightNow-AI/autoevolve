# Kernel frontier campaign

## Goal

This campaign studies add and add-with-scale vector kernels at two fixture sizes.
Each cell is selected with `AUTOEVOLVE_CELL`: `add-1k` and `add-8k` use `alpha=1.0`,
while `scale-1k` uses `alpha=0.375` and `scale-8k` uses `alpha=-1.25`. The suffix
selects 1024 or 8192 elements. The evaluator rejects unknown cells and evaluates all
four groups only when the variable is unset. The evaluator path is the bundled Triton
kernel evaluator.

## Method

The correctness gate runs before a score counts. The proxy budget is 20 child
evaluations per cell. The full budget is 200 child evaluations per cell. Every run
uses the tag `campaign:kernel-frontier:<cell>`.

## Promotion ladder

The first stage is a proxy candidate. The second stage requires improving proxy runs
from three distinct seeds. The scaled stage requires an explicit scaled validation
run. The campaign runner never claims the scaled stage automatically.

## Honesty

Without a compatible GPU, every returned score metric has a `mock_` prefix. A mock
metric is only a deterministic scheduling signal. It is not a performance claim.
GPU performance can be reported only from a real run with its exact run id.
