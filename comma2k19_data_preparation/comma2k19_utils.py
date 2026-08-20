from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


POSE_FILES = {
    "frame_times": "global_pose/frame_times",
    "frame_positions": "global_pose/frame_positions",
    "frame_velocities": "global_pose/frame_velocities",
    "frame_orientations": "global_pose/frame_orientations",
}

STATE_FEATURE_NAMES = (
    "longitudinal_position_m",
    "lateral_position_m",
    "longitudinal_velocity_mps",
    "lateral_velocity_mps",
    "speed_mps",
    "acceleration_mps2",
    "relative_heading_rad",
    "yaw_rate_radps",
)

TARGET_FEATURE_NAMES = (
    "longitudinal_position_m",
    "lateral_position_m",
    "speed_mps",
    "acceleration_mps2",
    "relative_heading_rad",
)


def load_array(path: Path, mmap_mode: str | None = "r"):
    if not path.exists():
        return None
    return np.load(path, mmap_mode=mmap_mode)


def load_pose_arrays(segment_path: Path):
    arrays = {}
    for name, relative_path in POSE_FILES.items():
        array = load_array(segment_path / relative_path)
        if array is None:
            raise FileNotFoundError(f"Missing {name}: {segment_path / relative_path}")
        arrays[name] = array
    return arrays


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalise(vector: np.ndarray, fallback: np.ndarray):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        vector = fallback
        norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot construct a local coordinate axis from a zero-length vector")
    return vector / norm


def _local_basis(positions, velocities, reference_index: int, history_start: int):
    reference_position = np.asarray(positions[reference_index], dtype=np.float64)
    up = _normalise(reference_position, np.array([0.0, 0.0, 1.0]))

    forward = np.asarray(velocities[reference_index], dtype=np.float64)
    forward = forward - np.dot(forward, up) * up
    fallback = np.asarray(positions[reference_index] - positions[history_start], dtype=np.float64)
    fallback = fallback - np.dot(fallback, up) * up
    forward = _normalise(forward, fallback)

    lateral = _normalise(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    return reference_position, forward, lateral


def scalar_motion_features(frame_times, velocities):
    times = np.asarray(frame_times, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        acceleration = np.zeros_like(speed)
    else:
        acceleration = np.gradient(speed, times)
    return speed, acceleration


def extract_motion_window(
    arrays,
    history_start: int,
    history_end: int,
    target_start: int,
    target_end: int,
):
    times = np.asarray(arrays["frame_times"], dtype=np.float64)
    positions = np.asarray(arrays["frame_positions"], dtype=np.float64)
    velocities = np.asarray(arrays["frame_velocities"], dtype=np.float64)

    if not (0 <= history_start <= history_end < target_start <= target_end < len(times)):
        raise IndexError("Invalid history/target frame range")

    reference_position, forward, lateral = _local_basis(
        positions, velocities, history_end, history_start
    )
    combined = slice(history_start, target_end + 1)
    displacement = positions[combined] - reference_position
    local_longitudinal = displacement @ forward
    local_lateral = displacement @ lateral
    longitudinal_velocity = velocities[combined] @ forward
    lateral_velocity = velocities[combined] @ lateral
    relative_heading = np.unwrap(np.arctan2(lateral_velocity, longitudinal_velocity))

    speed, acceleration = scalar_motion_features(times, velocities)
    combined_times = times[combined]
    if len(combined_times) < 2 or np.any(np.diff(combined_times) <= 0):
        yaw_rate = np.zeros_like(relative_heading)
    else:
        yaw_rate = np.gradient(relative_heading, combined_times)

    history_length = history_end - history_start + 1
    target_offset = target_start - history_start
    target_length = target_end - target_start + 1
    history_slice = slice(0, history_length)
    target_slice = slice(target_offset, target_offset + target_length)

    state_history = np.column_stack(
        (
            local_longitudinal[history_slice],
            local_lateral[history_slice],
            longitudinal_velocity[history_slice],
            lateral_velocity[history_slice],
            speed[history_start : history_end + 1],
            acceleration[history_start : history_end + 1],
            relative_heading[history_slice],
            yaw_rate[history_slice],
        )
    ).astype(np.float32)

    future_target = np.column_stack(
        (
            local_longitudinal[target_slice],
            local_lateral[target_slice],
            speed[target_start : target_end + 1],
            acceleration[target_start : target_end + 1],
            relative_heading[target_slice],
        )
    ).astype(np.float32)

    return state_history, future_target


def decode_video_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    image_size: tuple[int, int] = (224, 224),
):
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires PyAV and Pillow. Install requirements.txt first."
        ) from exc

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid video frame range")

    decoded = []
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index < start_frame:
                continue
            if frame_index > end_frame:
                break
            image = frame.to_image().convert("RGB")
            if image.size != image_size:
                image = image.resize(image_size, Image.Resampling.BILINEAR)
            decoded.append(np.asarray(image, dtype=np.uint8))

    expected = end_frame - start_frame + 1
    if len(decoded) != expected:
        raise RuntimeError(
            f"Decoded {len(decoded)} frames from {video_path}, expected {expected}"
        )
    return np.stack(decoded)
