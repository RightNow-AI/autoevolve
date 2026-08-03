"""Generate the deterministic two-moons-like classification fixture."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

SEED = 314_159
TRAIN_SIZE = 96
VALIDATION_SIZE = 48
FIXTURE_DIR = Path(__file__).resolve().parent


def build_rows() -> dict[str, object]:
    """Return seeded train and validation rows."""

    rng = random.Random(SEED)
    rows: list[dict[str, float | int]] = []
    total = TRAIN_SIZE + VALIDATION_SIZE
    for index in range(total):
        label = index % 2
        angle = rng.uniform(0.0, math.pi)
        noise_x = rng.uniform(-0.08, 0.08)
        noise_y = rng.uniform(-0.08, 0.08)
        if label == 0:
            x1 = math.cos(angle) + noise_x
            x2 = math.sin(angle) + noise_y
        else:
            x1 = 1.0 - math.cos(angle) + noise_x
            x2 = 0.5 - math.sin(angle) + noise_y
        rows.append(
            {
                "label": label,
                "x1": round(x1, 9),
                "x2": round(x2, 9),
            }
        )
    rng.shuffle(rows)
    return {
        "seed": SEED,
        "train": rows[:TRAIN_SIZE],
        "validation": rows[TRAIN_SIZE:],
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Write byte-stable fixture JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data.json").write_bytes(_json_bytes(build_rows()))


if __name__ == "__main__":
    write_fixtures()

