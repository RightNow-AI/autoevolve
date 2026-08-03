"""Generate deterministic uniform and clustered bin-packing families."""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 424_242
CAPACITY = 100
INSTANCE_COUNT = 10
ITEM_COUNT = 24
FIXTURE_DIR = Path(__file__).resolve().parent


def _family(name: str, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    instances: list[dict[str, object]] = []
    for index in range(INSTANCE_COUNT):
        if name == "uniform":
            items = [rng.randint(5, 45) for _ in range(ITEM_COUNT)]
        else:
            items = [
                max(5, min(45, rng.choice((12, 38)) + rng.randint(-3, 3)))
                for _ in range(ITEM_COUNT)
            ]
        instances.append({"key": f"{name}-{index:02d}", "items": items})
    return {
        "capacity": CAPACITY,
        "family": name,
        "instances": instances,
        "seed": seed,
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Write both byte-stable instance families."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, name in enumerate(("uniform", "clustered")):
        payload = _family(name, SEED + offset)
        (output_dir / f"{name}.json").write_bytes(_json_bytes(payload))


if __name__ == "__main__":
    write_fixtures()

