import pytest

from driving_algorithm.waymo.tfrecord import iter_tfrecord
from tests.waymo_test_utils import write_tfrecord


def test_iter_tfrecord_reads_multiple_payloads(tmp_path):
    path = tmp_path / "sample.tfrecord"
    write_tfrecord(path, [b"first", b"second"])

    assert list(iter_tfrecord(path)) == [b"first", b"second"]


def test_iter_tfrecord_rejects_corrupt_payload_crc(tmp_path):
    path = tmp_path / "corrupt.tfrecord"
    write_tfrecord(path, [b"payload"])
    raw = bytearray(path.read_bytes())
    raw[12] ^= 0xFF
    path.write_bytes(raw)

    with pytest.raises(ValueError, match=r"record 0.*CRC"):
        list(iter_tfrecord(path))
