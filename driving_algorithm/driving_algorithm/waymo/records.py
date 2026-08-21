from __future__ import annotations

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
