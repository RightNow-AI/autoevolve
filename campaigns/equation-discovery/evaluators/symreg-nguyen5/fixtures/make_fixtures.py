"""Generate deterministic Nguyen-5 train and held-out fixtures."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

SEED = 161_803
TRAIN_SIZE = 60
HELDOUT_SIZE = 40
FIXTURE_DIR = Path(__file__).resolve().parent


def target(x: float) -> float:
    """Return the Nguyen-5 target."""

    return math.sin(x * x) * math.cos(x) - 1.0


def build_splits() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return seeded samples from the closed interval minus one to one."""

    rng = random.Random(SEED)
    rows: list[dict[str, float]] = []
    for _ in range(TRAIN_SIZE + HELDOUT_SIZE):
        x = round(rng.uniform(-1.0, 1.0), 9)
        rows.append({"x": x, "y": round(target(x), 12)})
    return rows[:TRAIN_SIZE], rows[TRAIN_SIZE:]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Write both byte-stable split files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train, heldout = build_splits()
    (output_dir / "train.json").write_bytes(_json_bytes({"data": train, "seed": SEED}))
    (output_dir / "heldout.json").write_bytes(
        _json_bytes({"data": heldout, "seed": SEED})
    )


if __name__ == "__main__":
    write_fixtures()

