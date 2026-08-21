from __future__ import annotations

import re

from waymo_open_dataset.protos import end_to_end_driving_data_pb2


def parse_e2e_record(payload: bytes):
    record = end_to_end_driving_data_pb2.E2EDFrame()
    record.ParseFromString(payload)
    return record


def front_image_bytes(record) -> bytes | None:
    for image in record.frame.images:
        if image.name == 1:
            return bytes(image.image)
    return None


_FRAME_NAME_PATTERN = re.compile(r"^(?P<route>.+)-(?P<index>\d{3})$")


def split_frame_name(frame_name: str) -> tuple[str, int | None]:
    """Return the observed E2E route prefix and native frame index."""
    match = _FRAME_NAME_PATTERN.fullmatch(frame_name)
    if match is None:
        return frame_name, None
    return match.group("route"), int(match.group("index"))
