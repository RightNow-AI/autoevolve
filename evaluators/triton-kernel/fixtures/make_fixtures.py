"""Generate deterministic JSON vector cases for the Triton evaluator."""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 65_537
CASES = ((1_024, 0.375), (8_192, -1.25))
FIXTURE_DIR = Path(__file__).resolve().parent


def build_cases() -> dict[str, object]:
    """Return stable float lists without requiring NumPy or Triton."""
    rng = random.Random(SEED)
    cases: list[dict[str, object]] = []
    for size, alpha in CASES:
        x = [round(rng.uniform(-2.0, 2.0), 7) for _ in range(size)]
        y = [round(rng.uniform(-2.0, 2.0), 7) for _ in range(size)]
        cases.append(
            {
                "name": f"vector-{size}",
                "size": size,
                "alpha": alpha,
                "x": x,
                "y": y,
            }
        )
    return {"seed": SEED, "cases": cases}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Rewrite cases.json with byte-stable content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases.json").write_bytes(_json_bytes(build_cases()))


if __name__ == "__main__":
    write_fixtures()

