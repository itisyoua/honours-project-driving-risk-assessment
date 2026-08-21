import io
import struct
from pathlib import Path

import google_crc32c
import numpy as np
from PIL import Image
from waymo_open_dataset.protos import end_to_end_driving_data_pb2


def masked_crc32c(payload: bytes) -> int:
    crc = google_crc32c.value(payload)
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord(path: Path, payloads: list[bytes]) -> None:
    with path.open("wb") as output:
        for payload in payloads:
            length_bytes = struct.pack("<Q", len(payload))
            output.write(length_bytes)
            output.write(struct.pack("<I", masked_crc32c(length_bytes)))
            output.write(payload)
            output.write(struct.pack("<I", masked_crc32c(payload)))


def jpeg_bytes(value: int) -> bytes:
    image = Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    return encoded.getvalue()


def make_e2e_payload(run_id: str, timestamp_micros: int, image_value: int) -> bytes:
    record = end_to_end_driving_data_pb2.E2EDFrame()
    record.frame.context.name = run_id
    record.frame.timestamp_micros = timestamp_micros
    image = record.frame.images.add()
    image.name = 1
    image.image = jpeg_bytes(image_value)

    past_x = np.linspace(-15.0, 0.0, 16, dtype=np.float32)
    record.past_states.pos_x.extend(past_x)
    record.past_states.pos_y.extend(np.zeros(16, dtype=np.float32))
    record.past_states.vel_x.extend(np.full(16, 4.0, dtype=np.float32))
    record.past_states.vel_y.extend(np.zeros(16, dtype=np.float32))
    record.past_states.accel_x.extend(np.zeros(16, dtype=np.float32))
    record.past_states.accel_y.extend(np.zeros(16, dtype=np.float32))

    future_x = np.arange(1, 21, dtype=np.float32)
    record.future_states.pos_x.extend(future_x)
    record.future_states.pos_y.extend(np.zeros(20, dtype=np.float32))
    record.future_states.pos_z.extend(np.zeros(20, dtype=np.float32))
    record.intent = end_to_end_driving_data_pb2.EgoIntent.GO_STRAIGHT
    return record.SerializeToString()
