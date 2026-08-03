"""Generate deterministic initial conditions for the lander evaluator."""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 424_242
FIXTURE_DIR = Path(__file__).resolve().parent
BASE_CASES = (
    (48.0, -8.0, 0.45, -1.2),
    (56.0, 7.5, -0.55, -1.8),
    (66.0, -11.0, 0.75, -0.9),
    (74.0, 10.0, -0.65, -2.2),
    (43.0, 4.0, -0.25, -0.7),
    (82.0, -5.5, 0.35, -2.6),
)


def build_scenarios() -> dict[str, object]:
    """Return six seeded, varied, and solvable initial conditions."""
    rng = random.Random(SEED)
    scenarios: list[dict[str, float | str]] = []
    for index, (altitude, x, vx, vy) in enumerate(BASE_CASES, start=1):
        scenarios.append(
            {
                "name": f"scenario-{index:02d}",
                "x": round(x + rng.uniform(-0.6, 0.6), 6),
                "y": round(altitude + rng.uniform(-1.5, 1.5), 6),
                "vx": round(vx + rng.uniform(-0.12, 0.12), 6),
                "vy": round(vy + rng.uniform(-0.18, 0.18), 6),
                "angle": round(rng.uniform(-0.07, 0.07), 6),
                "angular_velocity": round(rng.uniform(-0.025, 0.025), 6),
                "fuel": float(20 + (index - 1) % 3),
            }
        )
    return {"seed": SEED, "scenarios": scenarios}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Rewrite scenarios.json with byte-stable content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenarios.json").write_bytes(_json_bytes(build_scenarios()))


if __name__ == "__main__":
    write_fixtures()
