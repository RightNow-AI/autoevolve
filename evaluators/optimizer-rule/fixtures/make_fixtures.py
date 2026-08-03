"""Generate deterministic disjoint two-moons training and validation fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 314_159
TRAIN_SIZE = 96
VALIDATION_SIZE = 48
NOISE_STD = 0.08
FIXTURE_DIR = Path(__file__).resolve().parent


def build_rows() -> dict[str, object]:
    """Return seeded, disjoint training and validation rows."""

    rng = np.random.default_rng(SEED)
    total = TRAIN_SIZE + VALIDATION_SIZE
    labels = np.arange(total, dtype=np.int64) % 2
    angles = rng.uniform(0.0, math.pi, size=total)
    noise = rng.normal(0.0, NOISE_STD, size=(total, 2))
    features = np.empty((total, 2), dtype=np.float64)
    first_class = labels == 0
    second_class = ~first_class
    features[first_class, 0] = np.cos(angles[first_class])
    features[first_class, 1] = np.sin(angles[first_class])
    features[second_class, 0] = 1.0 - np.cos(angles[second_class])
    features[second_class, 1] = 0.5 - np.sin(angles[second_class])
    features += noise

    rows: list[dict[str, float | int]] = []
    for sample_id in rng.permutation(total):
        rows.append(
            {
                "id": int(sample_id),
                "label": int(labels[sample_id]),
                "x1": round(float(features[sample_id, 0]), 9),
                "x2": round(float(features[sample_id, 1]), 9),
            }
        )
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
