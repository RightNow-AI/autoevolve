"""Working proportional-derivative seed for the lunar lander task.

Candidate contract:
act(state: dict[str, float], t: float) returns (throttle, gimbal).
The state keys are x, y, vx, vy, angle, angular_velocity, and fuel.
"""

from __future__ import annotations

import math


# Evolution may change only the function body between the markers.
def act(state: dict[str, float], t: float) -> tuple[float, float]:
    """Track a safe descent profile while damping horizontal and angular motion."""
    # EVOLVE-BLOCK-START
    del t
    altitude = max(state["y"], 0.0)
    target_vy = -min(3.6, 0.65 + 0.34 * math.sqrt(altitude))
    vertical_thrust = 1.62 + 0.85 * (target_vy - state["vy"])

    horizontal_accel = -0.045 * state["x"] - 0.32 * state["vx"]
    horizontal_accel = max(-0.9, min(0.9, horizontal_accel))
    desired_angle = math.atan2(horizontal_accel, max(vertical_thrust, 0.8))
    desired_angle = max(-0.28, min(0.28, desired_angle))

    gimbal = (
        2.2 * (desired_angle - state["angle"])
        - 1.15 * state["angular_velocity"]
    )
    required_thrust = math.hypot(max(vertical_thrust, 0.0), horizontal_accel)
    throttle = required_thrust / 6.0
    return throttle, gimbal
    # EVOLVE-BLOCK-END
