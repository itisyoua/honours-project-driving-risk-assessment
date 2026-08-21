import csv

from driving_algorithm.waymo.convert_records import convert_shards
from tests.waymo_test_utils import make_e2e_payload, write_tfrecord


def test_converter_writes_one_sample_per_native_record(tmp_path):
    shard = tmp_path / "validation.tfrecord-00000"
    write_tfrecord(
        shard,
        [
            make_e2e_payload(f"route-a-{index:03d}", 0, index * 30)
            for index in range(3)
        ],
    )
    output_root = tmp_path / "converted"
    manifest_path = tmp_path / "manifests" / "waymo_validation.csv"

    summary = convert_shards(
        [shard],
        output_root=output_root,
        manifest_path=manifest_path,
        split="validation",
    )

    assert summary.records == 3
    assert summary.samples == 3
    assert summary.routes == 1
    assert summary.rejected == 0
    assert len(list((output_root / "images").glob("*.jpg"))) == 3
    assert len(list((output_root / "motion").glob("*.npz"))) == 3
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert [row["frame_id"] for row in rows] == [
        "route-a-000",
        "route-a-001",
        "route-a-002",
    ]
    assert {row["route_id"] for row in rows} == {"route-a"}

