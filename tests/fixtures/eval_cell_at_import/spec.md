# Cell selected at import

This evaluator reads its cell at import time, before any candidate code can
load, so a candidate cannot choose the instance it is judged against. That is
what docs/FRONTIER.md requires of a frontier pack. Describing this evaluator
therefore only works if the describe probe is given the same workload
configuration the sandbox gets.
