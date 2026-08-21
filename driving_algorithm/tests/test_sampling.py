import pytest
import torch

from driving_algorithm.training.sampling import (
    source_balanced_sample_weights,
    validate_training_rows,
)


def test_training_rows_reject_non_training_split():
    rows = [
        {"route_id": "route-a", "split": "train", "source": "comma2k19"},
        {"route_id": "route-b", "split": "validation", "source": "waymo_e2e"},
    ]

    with pytest.raises(ValueError, match="only train rows"):
        validate_training_rows(rows)


def test_source_balanced_weights_equalise_source_probability():
    rows = [
        {"split": "train", "source": "comma2k19"},
        {"split": "train", "source": "comma2k19"},
        {"split": "train", "source": "comma2k19"},
        {"split": "train", "source": "waymo_e2e"},
    ]

    weights = source_balanced_sample_weights(rows)

    torch.testing.assert_close(
        weights[:3], torch.full((3,), 1.0 / 3.0, dtype=torch.double)
    )
    torch.testing.assert_close(weights[3:], torch.ones(1, dtype=torch.double))


def test_source_balanced_weights_support_single_source_manifest():
    rows = [{"split": "train"}, {"split": "train"}]

    weights = source_balanced_sample_weights(rows, default_source="waymo_e2e")

    torch.testing.assert_close(weights, torch.full((2,), 0.5, dtype=torch.double))
