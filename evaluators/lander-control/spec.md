# Lander control evaluator

## Task

The candidate implements a control policy for a two-dimensional lunar lander. On each
fixed simulation step, `act(state, t)` returns throttle and gimbal commands. Throttle is
clamped to `[0, 1]`. Gimbal is clamped to `[-1, 1]`. Non-finite controls fail closed.

The evaluator owns the simulator and all physics. Candidates cannot replace the dynamics or
report their own score. The state is `(x, y, vx, vy, angle, angular_velocity, fuel)`. The
simulator uses a fixed `0.05` second timestep, lunar gravity of `1.62 m/s^2`, and semi-implicit
Euler integration. Maximum thrust acceleration is `6.0 m/s^2`. Gimbal affects thrust direction
within `0.16` radians. Full-throttle gimbal authority is `2.4 rad/s^2`, with passive angular
damping of `0.55 /s`. Fuel burns at one unit per second at full throttle. Simulation time is
capped at `60` seconds.

## Gate and metrics

The `landed` gate requires every selected scenario to touch down with all of these conditions:

- `|vy| <= 2.0 m/s`
- `|vx| <= 1.0 m/s`
- `|angle| <= 0.2 rad`
- fuel does not run out before touchdown
- touchdown occurs within `60` simulated seconds

Any failure raises `EvalError` with the scenario name, failed condition, and measured value.
The committed `crash` mutant cuts throttle and fails the vertical touchdown-speed condition.

The headline metric is `fuel_efficiency`. It is
`(total_initial_fuel - total_fuel_used) / total_initial_fuel` across the selected scenarios.
The evaluator computes fuel use from its own simulation. The target semantics are maximize.
`mean_touchdown_speed` is the mean magnitude of touchdown velocity in meters per second.
`scenarios_landed` is the number of scenarios that passed the gate.

Stage 0 runs the first three scenarios and has a 15 second timeout. Stage 1 runs all six
scenarios and has a 30 second timeout. `ceiling()` returns `None`. A true fuel-optimal solution
would require solving the optimal control problem, which this pack does not do, so no ceiling is
claimed.

## Baseline controller

The seed is a working proportional-derivative controller. Its vertical loop tracks an
altitude-dependent descent profile that approaches a gentle terminal speed. Gravity
feed-forward supplies the hover term, and vertical-speed error adds damping. A bounded
horizontal position and velocity correction produces a desired angle. A second
proportional-derivative loop tracks that angle and damps angular velocity. These bounded loops
leave thrust and attitude margin for every fixture rather than relying on one exact trajectory.

## Hardware and dependencies

This evaluator needs only a CPU, Python, and NumPy. It is deterministic and offline. It uses no
network, wall clock, or simulation-time randomness in a gate decision.

## Fixture provenance

`fixtures/make_fixtures.py` uses Python `random.Random` seed `424242`. It generates six initial
conditions with varied altitude, horizontal offset, horizontal and vertical velocity, angle,
angular velocity, and fuel. The first three form stage 0. All six form stage 1.

Regenerate the committed file with:

```text
python evaluators/lander-control/fixtures/make_fixtures.py
```

The script rewrites byte-identical JSON for unchanged code.

## Certificate scope

This pack produces a simulated certificate. Results are exactly reproducible statements about
this simulator and these committed scenarios. They are not claims about real flight hardware,
unmodeled dynamics, sensors, actuators, or environmental conditions.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` in
`policy.py`. They must preserve `act(state: dict[str, float], t: float) -> tuple[float, float]`.
