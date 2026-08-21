import json

from driving_algorithm.waymo.inspect_records import inspect_shards
from tests.waymo_test_utils import make_e2e_payload, write_tfrecord


def test_inspector_summarises_waymo_e2e_records(tmp_path):
    shard = tmp_path / "validation.tfrecord-00000"
    payloads = [
        make_e2e_payload(f"run-1-{index:03d}", 0, index * 20)
        for index in range(6)
    ]
    write_tfrecord(shard, payloads)
    report_path = tmp_path / "report.json"

    report = inspect_shards([shard], report_path)

    assert report["records"] == 6
    assert report["routes"] == 1
    assert report["unique_frame_ids"] == 6
    assert report["front_camera_records"] == 6
    assert report["past_length_counts"] == {"16": 6}
    assert report["future_length_counts"] == {"20": 6}
    assert report["zero_timestamp_records"] == 6
    assert report["image_dimensions"] == {"32x24": 6}
    assert report["compatible_with_cnn_lstm_sample"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_inspector_rejects_record_without_front_image(tmp_path):
    shard = tmp_path / "training.tfrecord-00000"
    payload = make_e2e_payload("run-1-000", 0, 0)
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2

    record = end_to_end_driving_data_pb2.E2EDFrame.FromString(payload)
    del record.frame.images[:]
    write_tfrecord(shard, [record.SerializeToString()])

    report = inspect_shards([shard], tmp_path / "report.json")

    assert report["compatible_with_cnn_lstm_sample"] is False
    assert report["front_camera_records"] == 0
    assert report["malformed_records"] == 0
