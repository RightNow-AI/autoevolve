"""Keep the builtins table intact across candidate execution.

Evaluators in this project bind the primitives they depend on at module import
time, before any candidate is loaded:

    _LEN = len
    _CALLABLE = callable

That defends the gate's own direct calls. A candidate that assigns over
``builtins.len`` cannot change what ``_LEN`` already points at.

It does not defend calls the gate makes through the standard library. Those
functions resolve their own names from ``builtins`` at call time, so they see
whatever the candidate left behind. ``inspect.signature`` is the live example:
it calls ``callable(obj)`` internally while deciding how to introspect, and a
candidate that replaces ``builtins.callable`` with a function that raises will
take down signature detection in every pack that inspects a candidate
entrypoint. The trusted-primitive pattern reads as though it closed this and it
only closed half of it.

The general fix is to put the table back. Executing candidate code inside
``restored_builtins()`` snapshots ``builtins.__dict__`` first and restores it
afterwards, including names the candidate deleted or invented, so nothing a
candidate writes there outlives its own execution.

This is defence in depth rather than the sandbox. Candidates run in a
subprocess with resource limits and no network, which is what actually contains
them. This closes a hole inside that subprocess, where the gate and the
candidate share one interpreter and the gate has to keep working well enough to
return an honest verdict.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def restored_builtins() -> Iterator[None]:
    """Run a block, then restore ``builtins`` to its state on entry.

    Restoration runs in a ``finally``, so it also covers a candidate that
    raises partway through rewriting the table.
    """

    snapshot = dict(builtins.__dict__)
    try:
        yield
    finally:
        current = builtins.__dict__
        # Delete first: a candidate can add names as well as replace them, and
        # a leftover name is still a change to the table the gate runs on.
        for name in [name for name in current if name not in snapshot]:
            del current[name]
        current.update(snapshot)
