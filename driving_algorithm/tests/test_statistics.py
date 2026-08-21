import csv

import numpy as np
import torch

from driving_algorithm.data.statistics import compute_state_statistics
from driving_algorithm.runtime import select_device


def test_state_statistics_use_valid_history_steps(tmp_path):
    root = tmp_path / "converted"
    root.mkdir()
    rows = []
    for index, value in enumerate((1.0, 3.0)):
        path = root / f"motion-{index}.npz"
        state = np.full((16, 8), value, dtype=np.float32)
        mask = np.ones(16, dtype=np.bool_)
        np.savez_compressed(path, state_history=state, history_mask=mask)
        rows.append({"motion_path": path.name})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["motion_path"])
        writer.writeheader()
        writer.writerows(rows)

    statistics = compute_state_statistics(manifest, root)

    assert statistics.count == 32
    np.testing.assert_allclose(statistics.mean, np.full(8, 2.0))
    np.testing.assert_allclose(statistics.std, np.full(8, 1.0))


def test_explicit_cpu_device_is_respected():
    assert select_device("cpu") == torch.device("cpu")
