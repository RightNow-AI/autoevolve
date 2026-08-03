"""Generate deterministic image fixtures and their baseline outputs."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path
from types import SimpleNamespace

SEED = 104_729
SIZES = (24, 32, 48, 64, 80, 96)
FIXTURE_DIR = Path(__file__).resolve().parent
BASELINE_PATH = FIXTURE_DIR.parent / "baseline" / "pipeline.py"


def _load_baseline() -> SimpleNamespace:
    namespace = runpy.run_path(str(BASELINE_PATH), run_name="python_speedup_fixture_baseline")
    return SimpleNamespace(**namespace)


def build_images() -> dict[str, object]:
    """Return six seeded synthetic images with stable decimal coordinates."""
    rng = random.Random(SEED)
    images: list[dict[str, object]] = []
    for image_index, size in enumerate(SIZES):
        pixels: list[list[float]] = []
        for row in range(size):
            pixel_row: list[float] = []
            for column in range(size):
                lattice = ((7 * row + 11 * column + 13 * image_index) % 31) / 30.0
                checker = 0.18 if (row // 4 + column // 5 + image_index) % 2 else -0.18
                noise = (rng.random() - 0.5) * 0.08
                value = min(1.0, max(0.0, lattice + checker + noise))
                pixel_row.append(round(value, 9))
            pixels.append(pixel_row)
        images.append({"name": f"synthetic-{size}", "size": size, "pixels": pixels})
    return {"seed": SEED, "images": images}


def build_expected(images: dict[str, object]) -> dict[str, object]:
    """Compute exact threshold-count outputs with the bundled baseline."""
    baseline = _load_baseline()
    outputs: list[dict[str, object]] = []
    raw_images = images["images"]
    if not isinstance(raw_images, list):
        raise TypeError("images must be a list")
    for case in raw_images:
        if not isinstance(case, dict):
            raise TypeError("image case must be an object")
        pixels = case["pixels"]
        blurred = baseline.box_blur(pixels)
        magnitudes = baseline.sobel_magnitude(blurred)
        count = baseline.threshold_count(magnitudes)
        outputs.append({"name": case["name"], "value": count})
    return {"seed": SEED, "outputs": outputs}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_fixtures(output_dir: Path = FIXTURE_DIR) -> None:
    """Rewrite both fixture files with byte-stable content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = build_images()
    expected = build_expected(images)
    (output_dir / "images.json").write_bytes(_json_bytes(images))
    (output_dir / "expected.json").write_bytes(_json_bytes(expected))
    print(_json_bytes(expected).decode(), end="")


if __name__ == "__main__":
    write_fixtures()
