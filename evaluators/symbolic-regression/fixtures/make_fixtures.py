"""Generate deterministic Nguyen-7 train and held-out fixtures."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

SEED = 271_828
TRAIN_SIZE = 60
HELDOUT_SIZE = 40
FIXTURE_DIR = Path(__file__).resolve().parent


def target(x: float) -> float:
    """Return the published Nguyen-7 target value."""
    return math.log(x + 1.0) + math.log(x * x + 1.0)


def build_splits() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return seeded train and held-out samples from the interval zero to two."""
    rng = random.Random(SEED)
    rows: list[dict[str, float]] = []
    for _ in range(TRAIN_SIZE + HELDOUT_SIZE):
        x = round(rng.uniform(0.0, 2.0), 9)
        rows.append({"x": x, "y": round(target(x), 12)})
    return rows[:TRAIN_SIZE], rows[TRAIN_SIZE:]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Rewrite both split files with byte-stable content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train, heldout = build_splits()
    (output_dir / "train.json").write_bytes(_json_bytes({"seed": SEED, "data": train}))
    (output_dir / "heldout.json").write_bytes(
        _json_bytes({"seed": SEED, "data": heldout})
    )


if __name__ == "__main__":
    write_fixtures()

