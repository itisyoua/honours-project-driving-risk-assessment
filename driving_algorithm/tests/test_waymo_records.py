import json

from driving_algorithm.waymo.inspect_records import inspect_shards
from tests.waymo_test_utils import make_e2e_payload, write_tfrecord


def test_inspector_summarises_waymo_e2e_records(tmp_path):
    shard = tmp_path / "validation.tfrecord-00000"
    payloads = [
        make_e2e_payload("run-1", index * 250_000, index * 20)
        for index in range(6)
    ]
    write_tfrecord(shard, payloads)
    report_path = tmp_path / "report.json"

    report = inspect_shards([shard], report_path)

    assert report["records"] == 6
    assert report["runs"] == 1
    assert report["front_camera_records"] == 6
    assert report["past_length_counts"] == {"16": 6}
    assert report["future_length_counts"] == {"20": 6}
    assert report["timestamp_gap_micros"]["median"] == 250_000
    assert report["image_dimensions"] == {"32x24": 6}
    assert report["compatible_with_16_frame_history"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_inspector_marks_sixteen_frame_run_compatible(tmp_path):
    shard = tmp_path / "training.tfrecord-00000"
    write_tfrecord(
        shard,
        [
            make_e2e_payload("run-1", index * 250_000, index)
            for index in range(16)
        ],
    )

    report = inspect_shards([shard], tmp_path / "report.json")

    assert report["compatible_with_16_frame_history"] is True
    assert report["malformed_records"] == 0
