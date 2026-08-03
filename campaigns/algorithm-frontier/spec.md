# Algorithm frontier campaign

## Goal

This campaign evolves one-dimensional bin-packing heuristics. The baseline is
first-fit. The cells select uniform or clustered committed instance families.

## Method

The exact gate requires every item index exactly once and rejects any bin above its
capacity. The measured metric is total `bins_used` across the selected family, and
lower is better. The proxy budget is 20 child evaluations.

## Promotion ladder

One completed run is a proxy candidate. Promotion requires improving runs from three
distinct seeds. Scaled validation requires an explicit run and is never inferred.

## Honesty

Results apply only to the committed instance family selected by the cell. A result
does not describe other bin-packing distributions. Every result includes its run id.

