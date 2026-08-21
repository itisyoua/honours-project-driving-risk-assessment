import torch

from driving_algorithm.data.contracts import SequenceContract
from driving_algorithm.data.waymo_dataset import WaymoE2EDataset
from driving_algorithm.waymo.convert_records import convert_shards
from tests.waymo_test_utils import make_e2e_payload, write_tfrecord


def test_dataset_loads_model_ready_waymo_sample(tmp_path):
    shard = tmp_path / "validation.tfrecord-00000"
    write_tfrecord(shard, [make_e2e_payload("route-a-007", 0, 90)])
    output_root = tmp_path / "converted"
    manifest_path = tmp_path / "manifest.csv"
    convert_shards(
        [shard],
        output_root=output_root,
        manifest_path=manifest_path,
        split="validation",
    )

    sample = WaymoE2EDataset(manifest_path, output_root)[0]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["state_history"].shape == (16, 8)
    assert sample["future_target"].shape == (20, 5)
    assert all(
        torch.isfinite(sample[key]).all()
        for key in ("image", "state_history", "future_target")
    )
    assert sample["source"] == "waymo_e2e"
    assert sample["route_id"] == "route-a"
    SequenceContract.validate(sample)
