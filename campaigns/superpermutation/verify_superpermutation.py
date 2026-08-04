"""Independently verify a superpermutation, sharing no code with the pack.

A superpermutation on n symbols is a string containing every one of the n!
permutations as a contiguous substring. This reads a candidate's builder,
runs it, and checks the result from scratch.

For n = 6 the standard construction gives 873, the best published result is
872, and the best published lower bound is 867. So 872 matches a record and
871 or shorter would beat one.

Usage: verify_superpermutation.py <builder.py> <n>
"""

from __future__ import annotations

import importlib.util
import sys
from itertools import permutations
from pathlib import Path


def load_string(builder: Path, n: int) -> str:
    spec = importlib.util.spec_from_file_location("candidate_builder", builder)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {builder}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("build", "superpermutation", "construct", "make"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return str(fn(n))
            except TypeError:
                return str(fn())
    raise SystemExit(f"no builder entry point found in {builder}")


builder = Path(sys.argv[1])
n = int(sys.argv[2])
text = load_string(builder, n)

symbols = sorted(set(text))
print(f"length          {len(text)}")
print(f"distinct symbols {len(symbols)}: {''.join(symbols)}")

missing = []
present = 0
for perm in permutations(symbols[:n]):
    if "".join(perm) in text:
        present += 1
    else:
        missing.append("".join(perm))

expected = 1
for k in range(2, n + 1):
    expected *= k

print(f"permutations    {present} of {expected} present")
if missing:
    print(f"  MISSING {len(missing)}, first few: {missing[:5]}")
    print("VERDICT: NOT a superpermutation")
    raise SystemExit(1)

print("VERDICT: valid superpermutation")
records = {5: (153, 153), 6: (872, 867)}
if n in records:
    best, lower = records[n]
    if len(text) < best:
        print(f"  BEATS the published best of {best} for n={n}")
    elif len(text) == best:
        print(f"  MATCHES the published best of {best} for n={n}")
    else:
        print(f"  above the published best of {best} for n={n}")
    print(f"  published lower bound for n={n} is {lower}")
