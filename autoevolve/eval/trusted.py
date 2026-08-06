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
from types import TracebackType

# Bound now, for the same reason evaluators bind their own primitives: this
# module has to keep working while the table it repairs is broken.
_DICT = dict


class restored_builtins:  # noqa: N801
    """Run a block, then restore ``builtins`` to its state on entry.

    Deliberately a class rather than a ``@contextmanager`` generator. A
    generator-based manager is driven by ``contextlib``, whose ``__exit__``
    calls the builtin ``next`` to resume it, so a candidate that replaced
    ``next`` breaks the very cleanup meant to undo the damage. The ``with``
    statement looks ``__enter__`` and ``__exit__`` up as type slots and never
    goes through ``builtins`` at all, which is what makes this version safe.

    Restoration runs on every exit path, so it also covers a candidate that
    raises partway through rewriting the table.
    """

    __slots__ = ("_snapshot",)

    def __enter__(self) -> None:
        self._snapshot = _DICT(builtins.__dict__)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        snapshot = self._snapshot
        current = builtins.__dict__
        # Delete first: a candidate can add names as well as replace them, and
        # a leftover name is still a change to the table the gate runs on.
        # The comprehension is materialised before deleting, because mutating
        # a dict while iterating it raises.
        for name in [name for name in current if name not in snapshot]:
            del current[name]
        current.update(snapshot)
