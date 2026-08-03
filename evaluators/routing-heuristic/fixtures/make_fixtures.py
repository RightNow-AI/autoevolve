"""Generate deterministic Euclidean routing instances."""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 314_159
SIZES = (30, 40, 50, 60, 75, 90, 105, 120)
FIXTURE_DIR = Path(__file__).resolve().parent


def build_instances() -> dict[str, object]:
    """Return eight seeded uniform point sets with stable decimal coordinates."""
    rng = random.Random(SEED)
    instances: list[dict[str, object]] = []
    for size in SIZES:
        points = [
            [round(rng.uniform(0.0, 1_000.0), 9), round(rng.uniform(0.0, 1_000.0), 9)]
            for _ in range(size)
        ]
        instances.append({"name": f"uniform-{size}", "size": size, "points": points})
    return {"seed": SEED, "instances": instances}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Rewrite instances.json with byte-stable content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "instances.json").write_bytes(_json_bytes(build_instances()))


if __name__ == "__main__":
    write_fixtures()

