"""Deliberately direct pure-Python image processing seed.

Candidate contract:
box_blur(image) returns the 3 by 3 box-blurred interior as rows of floats.
sobel_magnitude(image) returns the Sobel magnitude interior as rows of floats.
threshold_count(image, threshold=0.35) returns the number of values above the threshold.
"""

from __future__ import annotations

import math

Image = list[list[float]]


# EVOLVE-BLOCK-START
def box_blur(image: Image) -> Image:
    """Apply a 3 by 3 box blur using intentionally simple nested loops."""
    height = len(image)
    width = len(image[0])
    blurred: Image = []
    for row in range(1, height - 1):
        output_row: list[float] = []
        for column in range(1, width - 1):
            total = 0.0
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    total += image[row + row_offset][column + column_offset]
            output_row.append(total / 9.0)
        blurred.append(output_row)
    return blurred


def sobel_magnitude(image: Image) -> Image:
    """Compute Sobel gradient magnitudes using intentionally simple loops."""
    height = len(image)
    width = len(image[0])
    magnitudes: Image = []
    for row in range(1, height - 1):
        output_row: list[float] = []
        for column in range(1, width - 1):
            gx = (
                -image[row - 1][column - 1]
                + image[row - 1][column + 1]
                - 2.0 * image[row][column - 1]
                + 2.0 * image[row][column + 1]
                - image[row + 1][column - 1]
                + image[row + 1][column + 1]
            )
            gy = (
                image[row - 1][column - 1]
                + 2.0 * image[row - 1][column]
                + image[row - 1][column + 1]
                - image[row + 1][column - 1]
                - 2.0 * image[row + 1][column]
                - image[row + 1][column + 1]
            )
            output_row.append(math.sqrt(gx * gx + gy * gy))
        magnitudes.append(output_row)
    return magnitudes


def threshold_count(image: Image, threshold: float = 0.35) -> int:
    """Count gradient magnitudes strictly above the fixed decision threshold."""
    count = 0
    for row in image:
        for value in row:
            if value > threshold:
                count += 1
    return count
# EVOLVE-BLOCK-END

