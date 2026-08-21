import numpy as np
import pytest

from driving_algorithm.data.contracts import SequenceContract, make_sample_id


def valid_sample():
    return {
        "frames": np.zeros((16, 3, 224, 224), dtype=np.float32),
        "state_history": np.zeros((16, 8), dtype=np.float32),
        "future_target": np.zeros((20, 5), dtype=np.float32),
        "history_mask": np.ones(16, dtype=np.bool_),
        "future_mask": np.ones(20, dtype=np.bool_),
        "sample_id": "waymo_e2e:run-1:1000000",
        "route_id": "run-1",
        "source": "waymo_e2e",
        "split": "train",
        "scene_type": "go_straight",
    }


def test_contract_accepts_expected_shapes():
    SequenceContract.validate(valid_sample())


def test_contract_rejects_non_finite_motion():
    sample = valid_sample()
    sample["future_target"][2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SequenceContract.validate(sample)


def test_contract_rejects_wrong_frame_shape():
    sample = valid_sample()
    sample["frames"] = np.zeros((15, 3, 224, 224), dtype=np.float32)
    with pytest.raises(ValueError, match="frames"):
        SequenceContract.validate(sample)


def test_sample_id_is_source_qualified():
    assert make_sample_id("waymo_e2e", "run-1", 1_000_000) == (
        "waymo_e2e:run-1:1000000"
    )
