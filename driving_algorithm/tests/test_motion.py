import numpy as np
import pytest

from driving_algorithm.data.motion import (
    derive_future_target,
    derive_state_history,
    validate_timestamps,
)


def test_straight_history_has_expected_speed_heading_and_origin():
    x = np.arange(16, dtype=np.float32) * 2.5
    y = np.zeros(16, dtype=np.float32)
    vx = np.full(16, 10.0, dtype=np.float32)
    vy = np.zeros(16, dtype=np.float32)
    ax = np.zeros(16, dtype=np.float32)
    ay = np.zeros(16, dtype=np.float32)

    state = derive_state_history(x, y, vx, vy, ax, ay, dt=0.25)

    assert state.shape == (16, 8)
    np.testing.assert_allclose(state[-1, :2], [0.0, 0.0])
    np.testing.assert_allclose(state[:, 4], 10.0)
    np.testing.assert_allclose(state[:, 5:], 0.0, atol=1e-6)


def test_turning_history_wraps_relative_heading_and_derives_yaw_rate():
    heading = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
    speed = np.full(16, 8.0, dtype=np.float32)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = np.cumsum(vx * 0.25)
    y = np.cumsum(vy * 0.25)

    state = derive_state_history(
        x,
        y,
        vx,
        vy,
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        dt=0.25,
    )

    assert state[-1, 6] == pytest.approx(0.0, abs=1e-6)
    assert state[:, 7].mean() == pytest.approx(
        (heading[-1] - heading[0]) / (15 * 0.25), rel=1e-4
    )


def test_future_target_uses_local_positions_and_five_features():
    x = np.arange(1, 21, dtype=np.float32) * 2.5
    y = np.zeros(20, dtype=np.float32)

    target = derive_future_target(
        x,
        y,
        initial_velocity_xy=np.array([10.0, 0.0], dtype=np.float32),
        dt=0.25,
    )

    assert target.shape == (20, 5)
    np.testing.assert_allclose(target[-1, :2], [50.0, 0.0])
    np.testing.assert_allclose(target[:, 2], 10.0)
    np.testing.assert_allclose(target[:, 3:], 0.0, atol=1e-6)


def test_timestamp_gap_is_rejected():
    timestamps = np.arange(16, dtype=np.int64) * 250_000
    timestamps[8:] += 100_000

    with pytest.raises(ValueError, match="timestamp"):
        validate_timestamps(timestamps)


def test_regular_timestamps_are_accepted():
    validate_timestamps(np.arange(16, dtype=np.int64) * 250_000)
