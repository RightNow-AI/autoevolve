"""Deterministic two-dimensional lunar lander control evaluator."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import NamedTuple, cast

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics


def _load_trusted_numpy() -> ModuleType:
    """Import NumPy without allowing the candidate working directory to shadow it."""
    original_path = sys.path[:]
    working_directory = Path.cwd().resolve()
    safe_path: list[str] = []
    for entry in original_path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            safe_path.append(entry)
            continue
        if resolved != working_directory:
            safe_path.append(entry)
    sys.path[:] = safe_path
    try:
        return importlib.import_module("numpy")
    finally:
        sys.path[:] = original_path


np = _load_trusted_numpy()

STAGES: list[StageSpec] = [
    StageSpec(name="three-scenarios", timeout_s=15.0),
    StageSpec(name="all-scenarios", timeout_s=30.0),
]
GATE: str = "landed"
METRIC: str = "fuel_efficiency"
MAXIMIZE: bool = True

PACK_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PACK_DIR / "fixtures" / "scenarios.json"

DT_S = 0.05
GRAVITY_MPS2 = 1.62
MAX_THRUST_MPS2 = 6.0
MAX_GIMBAL_ANGLE_RAD = 0.16
MAX_ANGULAR_ACCEL_RAD_S2 = 2.4
ANGULAR_DAMPING_PER_S = 0.55
FUEL_BURN_PER_S = 1.0
TIME_LIMIT_S = 60.0
MAX_TOUCHDOWN_VY_MPS = 2.0
MAX_TOUCHDOWN_VX_MPS = 1.0
MAX_TOUCHDOWN_ANGLE_RAD = 0.2

_X = 0
_Y = 1
_VX = 2
_VY = 3
_ANGLE = 4
_ANGULAR_VELOCITY = 5
_FUEL = 6

_ARRAY = np.array
_FLOAT64 = np.float64
_SIN = np.sin
_COS = np.cos
_HYPOT = np.hypot
_ISFINITE = np.isfinite

StateVector = np.ndarray
Policy = Callable[[dict[str, float], float], object]


class Scenario(NamedTuple):
    """One immutable evaluator-owned initial condition."""

    name: str
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    fuel: float

    def state_vector(self) -> StateVector:
        """Return a fresh simulator state for this scenario."""
        return _ARRAY(
            [
                self.x,
                self.y,
                self.vx,
                self.vy,
                self.angle,
                self.angular_velocity,
                self.fuel,
            ],
            dtype=_FLOAT64,
        )


class Touchdown(NamedTuple):
    """Evaluator-owned touchdown measurements for one scenario."""

    vx: float
    vy: float
    angle: float
    fuel: float


def _fixture_float(item: dict[str, object], field: str, scenario_name: str) -> float:
    value = item.get(field)
    if type(value) not in (int, float):
        raise EvalError(f"fixture scenario {scenario_name} field {field} must be numeric")
    numeric = float(value)
    if not bool(_ISFINITE(numeric)):
        raise EvalError(f"fixture scenario {scenario_name} field {field} must be finite")
    return numeric


def _load_scenarios() -> tuple[Scenario, ...]:
    try:
        raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvalError(f"could not read lander scenarios: {exc}") from exc
    if type(raw) is not dict:
        raise EvalError("lander fixture root must be an object")
    seed = raw.get("seed")
    if type(seed) is not int:
        raise EvalError("lander fixture seed must be an integer")
    items = raw.get("scenarios")
    if type(items) is not list or len(items) != 6:
        raise EvalError("lander fixtures must contain exactly six scenarios")

    scenarios: list[Scenario] = []
    names: set[str] = set()
    for index, raw_item in enumerate(items):
        if type(raw_item) is not dict:
            raise EvalError(f"fixture scenario {index} must be an object")
        item = cast(dict[str, object], raw_item)
        name = item.get("name")
        if type(name) is not str or not name:
            raise EvalError(f"fixture scenario {index} must have a non-empty name")
        if name in names:
            raise EvalError(f"fixture scenario name is duplicated: {name}")
        names.add(name)
        scenario = Scenario(
            name=name,
            x=_fixture_float(item, "x", name),
            y=_fixture_float(item, "y", name),
            vx=_fixture_float(item, "vx", name),
            vy=_fixture_float(item, "vy", name),
            angle=_fixture_float(item, "angle", name),
            angular_velocity=_fixture_float(item, "angular_velocity", name),
            fuel=_fixture_float(item, "fuel", name),
        )
        if scenario.y <= 0.0:
            raise EvalError(f"fixture scenario {name} altitude must be positive")
        if scenario.fuel <= 0.0:
            raise EvalError(f"fixture scenario {name} fuel must be positive")
        scenarios.append(scenario)
    return tuple(scenarios)


def _stage_scenarios(stage: int) -> tuple[Scenario, ...]:
    if type(stage) is not int or stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    scenarios = _load_scenarios()
    return scenarios[:3] if stage == 0 else scenarios


def _load_policy(candidate_dir: Path) -> Policy:
    entry_path = candidate_dir / "policy.py"
    if not entry_path.is_file():
        raise EvalError(f"candidate is missing {entry_path.name}")
    module_name = f"_autoevolve_lander_control_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    try:
        act = module.act
    except AttributeError as exc:
        raise EvalError("candidate policy.py must define callable act(state, t)") from exc
    if not callable(act):
        raise EvalError("candidate policy.py must define callable act(state, t)")
    return cast(Policy, act)


def _normalize_control(
    raw_control: object,
    scenario_name: str,
    time_s: float,
) -> tuple[float, float]:
    """Snapshot one candidate-controlled traversal into two built-in floats."""
    try:
        snapshot = tuple(cast(Iterable[object], raw_control))
    except Exception as exc:
        error_name = type(exc).__name__
        raise EvalError(
            f"scenario {scenario_name} returned unreadable control at "
            f"t={time_s:.2f} s ({error_name})"
        ) from exc
    if len(snapshot) != 2:
        raise EvalError(
            f"scenario {scenario_name} returned {len(snapshot)} controls at "
            f"t={time_s:.2f} s; expected 2"
        )
    if type(snapshot[0]) is bool or type(snapshot[1]) is bool:
        raise EvalError(
            f"scenario {scenario_name} returned boolean control at t={time_s:.2f} s"
        )
    try:
        controls = (float(snapshot[0]), float(snapshot[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvalError(
            f"scenario {scenario_name} returned non-numeric control at t={time_s:.2f} s"
        ) from exc
    throttle, gimbal = controls
    if not bool(_ISFINITE(throttle)):
        raise EvalError(
            f"scenario {scenario_name} returned non-finite throttle={throttle} "
            f"at t={time_s:.2f} s"
        )
    if not bool(_ISFINITE(gimbal)):
        raise EvalError(
            f"scenario {scenario_name} returned non-finite gimbal={gimbal} "
            f"at t={time_s:.2f} s"
        )
    return (
        min(1.0, max(0.0, throttle)),
        min(1.0, max(-1.0, gimbal)),
    )


def _policy_state(state: StateVector) -> dict[str, float]:
    return {
        "x": float(state[_X]),
        "y": float(state[_Y]),
        "vx": float(state[_VX]),
        "vy": float(state[_VY]),
        "angle": float(state[_ANGLE]),
        "angular_velocity": float(state[_ANGULAR_VELOCITY]),
        "fuel": float(state[_FUEL]),
    }


def _touchdown_or_error(scenario_name: str, state: StateVector) -> Touchdown:
    vx = float(state[_VX])
    vy = float(state[_VY])
    angle = float(state[_ANGLE])
    fuel = float(state[_FUEL])
    if abs(vy) > MAX_TOUCHDOWN_VY_MPS:
        raise EvalError(
            f"scenario {scenario_name} touchdown vertical speed failed: "
            f"|vy|={abs(vy):.6f} m/s exceeds {MAX_TOUCHDOWN_VY_MPS:.6f} m/s"
        )
    if abs(vx) > MAX_TOUCHDOWN_VX_MPS:
        raise EvalError(
            f"scenario {scenario_name} touchdown horizontal speed failed: "
            f"|vx|={abs(vx):.6f} m/s exceeds {MAX_TOUCHDOWN_VX_MPS:.6f} m/s"
        )
    if abs(angle) > MAX_TOUCHDOWN_ANGLE_RAD:
        raise EvalError(
            f"scenario {scenario_name} touchdown angle failed: "
            f"|angle|={abs(angle):.6f} rad exceeds {MAX_TOUCHDOWN_ANGLE_RAD:.6f} rad"
        )
    return Touchdown(vx=vx, vy=vy, angle=angle, fuel=fuel)


def _simulate(policy: Policy, scenario: Scenario) -> Touchdown:
    state = scenario.state_vector()
    step_count = int(round(TIME_LIMIT_S / DT_S))
    for step in range(step_count):
        time_s = step * DT_S
        try:
            raw_control = policy(_policy_state(state), time_s)
        except Exception as exc:
            error_name = type(exc).__name__
            raise EvalError(
                f"scenario {scenario.name} policy raised {error_name} at t={time_s:.2f} s"
            ) from exc
        throttle, gimbal = _normalize_control(raw_control, scenario.name, time_s)

        requested_burn = throttle * FUEL_BURN_PER_S * DT_S
        fuel_burn = min(float(state[_FUEL]), requested_burn)
        effective_throttle = fuel_burn / (FUEL_BURN_PER_S * DT_S)
        thrust_accel = MAX_THRUST_MPS2 * effective_throttle
        thrust_angle = float(state[_ANGLE]) + MAX_GIMBAL_ANGLE_RAD * gimbal
        acceleration_x = float(thrust_accel * _SIN(thrust_angle))
        acceleration_y = float(thrust_accel * _COS(thrust_angle) - GRAVITY_MPS2)
        angular_accel = (
            MAX_ANGULAR_ACCEL_RAD_S2 * effective_throttle * gimbal
            - ANGULAR_DAMPING_PER_S * float(state[_ANGULAR_VELOCITY])
        )

        state[_VX] += acceleration_x * DT_S
        state[_VY] += acceleration_y * DT_S
        state[_ANGULAR_VELOCITY] += angular_accel * DT_S
        state[_X] += state[_VX] * DT_S
        state[_Y] += state[_VY] * DT_S
        state[_ANGLE] += state[_ANGULAR_VELOCITY] * DT_S
        state[_FUEL] -= fuel_burn

        if state[_Y] <= 0.0:
            return _touchdown_or_error(scenario.name, state)
        if state[_FUEL] <= 0.0:
            raise EvalError(
                f"scenario {scenario.name} ran out of fuel before touchdown: "
                f"fuel={float(state[_FUEL]):.6f} at t={(step + 1) * DT_S:.2f} s"
            )

    raise EvalError(
        f"scenario {scenario.name} time limit failed: "
        f"time={TIME_LIMIT_S:.2f} s, y={float(state[_Y]):.6f} m"
    )


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate safe touchdowns, then score evaluator-computed remaining fuel."""
    scenarios = _stage_scenarios(stage)
    policy = _load_policy(candidate_dir)
    touchdowns = [_simulate(policy, scenario) for scenario in scenarios]

    total_initial_fuel = sum(scenario.fuel for scenario in scenarios)
    total_fuel_used = sum(
        scenario.fuel - touchdown.fuel
        for scenario, touchdown in zip(scenarios, touchdowns, strict=True)
    )
    fuel_efficiency = (total_initial_fuel - total_fuel_used) / total_initial_fuel
    mean_touchdown_speed = sum(
        float(_HYPOT(touchdown.vx, touchdown.vy)) for touchdown in touchdowns
    ) / len(touchdowns)
    return {
        GATE: 1.0,
        METRIC: fuel_efficiency,
        "mean_touchdown_speed": mean_touchdown_speed,
        "scenarios_landed": float(len(touchdowns)),
        **source_metrics(candidate_dir, "policy.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no claimed fuel optimum for this control problem."""
    return None


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
