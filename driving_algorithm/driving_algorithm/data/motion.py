from __future__ import annotations

import numpy as np

from .contracts import FUTURE_STEPS, HISTORY_STEPS


def _as_vector(name: str, values, expected_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_length,):
        raise ValueError(
            f"{name} must have shape ({expected_length},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def derive_state_history(
    pos_x,
    pos_y,
    vel_x,
    vel_y,
    accel_x,
    accel_y,
    dt: float,
) -> np.ndarray:
    """Build the eight ego-history features in the current ego-local frame."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive value")

    x = _as_vector("pos_x", pos_x, HISTORY_STEPS)
    y = _as_vector("pos_y", pos_y, HISTORY_STEPS)
    vx = _as_vector("vel_x", vel_x, HISTORY_STEPS)
    vy = _as_vector("vel_y", vel_y, HISTORY_STEPS)
    ax = _as_vector("accel_x", accel_x, HISTORY_STEPS)
    ay = _as_vector("accel_y", accel_y, HISTORY_STEPS)

    local_x = x - x[-1]
    local_y = y - y[-1]
    speed = np.hypot(vx, vy)
    acceleration = np.zeros_like(speed)
    moving = speed > 1e-6
    acceleration[moving] = (
        ax[moving] * vx[moving] + ay[moving] * vy[moving]
    ) / speed[moving]

    heading = np.unwrap(np.arctan2(vy, vx))
    relative_heading = _wrap_angle(heading - heading[-1])
    yaw_rate = np.gradient(heading, dt)

    state = np.column_stack(
        (
            local_x,
            local_y,
            vx,
            vy,
            speed,
            acceleration,
            relative_heading,
            yaw_rate,
        )
    )
    return state.astype(np.float32)


def derive_future_target(
    pos_x,
    pos_y,
    initial_velocity_xy,
    dt: float,
) -> np.ndarray:
    """Build future x/y, speed, acceleration and relative heading targets."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive value")

    x = _as_vector("pos_x", pos_x, FUTURE_STEPS)
    y = _as_vector("pos_y", pos_y, FUTURE_STEPS)
    initial_velocity = np.asarray(initial_velocity_xy, dtype=np.float64)
    if initial_velocity.shape != (2,) or not np.isfinite(initial_velocity).all():
        raise ValueError("initial_velocity_xy must contain two finite values")

    vx = np.empty(FUTURE_STEPS, dtype=np.float64)
    vy = np.empty(FUTURE_STEPS, dtype=np.float64)
    vx[0], vy[0] = initial_velocity
    vx[1:] = np.diff(x) / dt
    vy[1:] = np.diff(y) / dt

    speed = np.hypot(vx, vy)
    initial_speed = float(np.hypot(*initial_velocity))
    acceleration = np.diff(np.concatenate(([initial_speed], speed))) / dt

    heading = np.unwrap(np.arctan2(vy, vx))
    initial_heading = float(np.arctan2(initial_velocity[1], initial_velocity[0]))
    relative_heading = _wrap_angle(heading - initial_heading)

    target = np.column_stack((x, y, speed, acceleration, relative_heading))
    return target.astype(np.float32)


def validate_timestamps(
    timestamps_micros,
    expected_dt_micros: int = 250_000,
    tolerance_micros: int = 25_000,
) -> None:
    timestamps = np.asarray(timestamps_micros, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps must be a one-dimensional sequence")
    if expected_dt_micros <= 0 or tolerance_micros < 0:
        raise ValueError("timestamp interval and tolerance must be valid")

    gaps = np.diff(timestamps)
    if np.any(gaps <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if np.any(np.abs(gaps - expected_dt_micros) > tolerance_micros):
        raise ValueError(
            "timestamp gaps must be within "
            f"{tolerance_micros} microseconds of {expected_dt_micros}"
        )
