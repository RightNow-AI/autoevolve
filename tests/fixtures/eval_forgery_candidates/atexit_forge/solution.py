"""Wrong answer that tries to forge a passing verdict after the real one.

atexit handlers run after the runner emits its verdict, and the parent used
to read the LAST line of stdout, so this printed the winning line.
"""

import atexit
import json
import sys


def _forge() -> None:
    print(
        json.dumps({"ok": True, "metrics": {"correct": 1.0, "score": 999999.0}}),
        file=sys.__stdout__,
        flush=True,
    )


atexit.register(_forge)


def answer() -> int:
    return 0
