from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import google_crc32c


_LENGTH = struct.Struct("<Q")
_CRC = struct.Struct("<I")
_MAX_RECORD_BYTES = 512 * 1024 * 1024


def _masked_crc32c(payload: bytes) -> int:
    crc = google_crc32c.value(payload)
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def _read_exact(stream, byte_count: int, record_index: int, field: str) -> bytes:
    payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError(
            f"record {record_index} is truncated while reading {field}: "
            f"expected {byte_count} bytes, got {len(payload)}"
        )
    return payload


def iter_tfrecord(path: Path, verify_crc: bool = True) -> Iterator[bytes]:
    """Yield raw records from a standard uncompressed TFRecord file."""
    path = Path(path)
    with path.open("rb") as stream:
        record_index = 0
        while True:
            length_bytes = stream.read(_LENGTH.size)
            if not length_bytes:
                return
            if len(length_bytes) != _LENGTH.size:
                raise ValueError(f"record {record_index} has a truncated length header")

            length_crc_bytes = _read_exact(
                stream, _CRC.size, record_index, "length CRC"
            )
            stored_length_crc = _CRC.unpack(length_crc_bytes)[0]
            if verify_crc and stored_length_crc != _masked_crc32c(length_bytes):
                raise ValueError(f"record {record_index} length CRC mismatch")

            length = _LENGTH.unpack(length_bytes)[0]
            if length > _MAX_RECORD_BYTES:
                raise ValueError(
                    f"record {record_index} length {length} exceeds safety limit"
                )
            payload = _read_exact(stream, length, record_index, "payload")
            payload_crc_bytes = _read_exact(
                stream, _CRC.size, record_index, "payload CRC"
            )
            stored_payload_crc = _CRC.unpack(payload_crc_bytes)[0]
            if verify_crc and stored_payload_crc != _masked_crc32c(payload):
                raise ValueError(f"record {record_index} payload CRC mismatch")

            yield payload
            record_index += 1
