import csv

from PIL import Image

from driving_algorithm.waymo.convert_records import convert_shards
from driving_algorithm.waymo.preview_sequence import preview_sample
from driving_algorithm.waymo.validate_preparation import validate_preparation
from tests.waymo_test_utils import make_e2e_payload, write_tfrecord


def converted_fixture(tmp_path):
    shard = tmp_path / "validation.tfrecord-00000"
    write_tfrecord(
        shard,
        [
            make_e2e_payload("route-a-001", 0, 30),
            make_e2e_payload("route-a-002", 0, 90),
        ],
    )
    output_root = tmp_path / "converted"
    manifest_path = tmp_path / "manifest.csv"
    convert_shards(
        [shard],
        output_root=output_root,
        manifest_path=manifest_path,
        split="validation",
    )
    return output_root, manifest_path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_validator_accepts_valid_conversion(tmp_path):
    output_root, manifest_path = converted_fixture(tmp_path)

    report = validate_preparation(manifest_path, output_root)

    assert report["errors"] == []
    assert report["samples"] == 2
    assert report["routes"] == 1
    assert report["split_counts"] == {"validation": 2}


def test_validator_reports_route_split_leakage(tmp_path):
    output_root, manifest_path = converted_fixture(tmp_path)
    rows = read_rows(manifest_path)
    rows[1]["split"] = "train"
    write_rows(manifest_path, rows)

    report = validate_preparation(manifest_path, output_root)

    assert any(error["code"] == "route_split_leakage" for error in report["errors"])


def test_validator_reports_exact_missing_image_path(tmp_path):
    output_root, manifest_path = converted_fixture(tmp_path)
    missing_relative = read_rows(manifest_path)[0]["image_path"]
    (output_root / missing_relative).unlink()

    report = validate_preparation(manifest_path, output_root)

    assert {
        "code": "missing_file",
        "sample_id": "waymo_e2e:route-a-001",
        "path": missing_relative,
    } in report["errors"]


def test_preview_writes_readable_png(tmp_path):
    output_root, manifest_path = converted_fixture(tmp_path)
    preview_path = tmp_path / "preview.png"

    preview_sample(manifest_path, output_root, preview_path, index=0)

    with Image.open(preview_path) as preview:
        assert preview.format == "PNG"
        assert preview.width >= 1000
        assert preview.height >= 400
