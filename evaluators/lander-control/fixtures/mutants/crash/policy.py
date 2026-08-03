"""Known-bad lander policy that never produces thrust."""

from __future__ import annotations


def act(state: dict[str, float], t: float) -> tuple[float, float]:
    """Cut the engine so the lander fails the touchdown-speed gate."""
    del state, t
    return 0.0, 0.0
