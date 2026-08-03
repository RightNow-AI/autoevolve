"""Seed Golomb ruler, valid by construction at every order.

build(order) returns a list of integer marks whose pairwise differences are
all distinct. The Erdos-Turan construction gives a Sidon set for any prime p:
the marks 2*p*k + (k*k mod p) for k in 0..p-1 have distinct pairwise
differences. Taking the first `order` of them keeps that property, because
any subset of a Sidon set is a Sidon set.

The seed is correct but far from optimal, which is the point: the whole gap
to the known optimum is headroom for evolution to close.
"""

from __future__ import annotations


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


# EVOLVE-BLOCK-START
def build(order: int) -> list[int]:
    """Return `order` marks forming a Golomb ruler, starting at 0."""

    prime = max(order, 2)
    while not _is_prime(prime):
        prime += 1
    marks = [2 * prime * k + (k * k) % prime for k in range(prime)]
    chosen = sorted(marks)[:order]
    base = chosen[0]
    return [value - base for value in chosen]
# EVOLVE-BLOCK-END
