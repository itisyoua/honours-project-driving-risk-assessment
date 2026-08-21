# Driving Algorithm

This package prepares source-independent driving samples for the honours
project's CNN-LSTM trajectory predictor. It runs on the Apple Silicon
development Mac without importing or running CARLA.

The common sample timeline is 4 Hz with 16 history steps (4 seconds) and 20
future steps (5 seconds). Raw and converted Waymo data live under `data/` and
are intentionally excluded from Git.

The unified baseline uses a CNN for the current front camera image and an LSTM
for the 16 provided ego-motion states. Their encoded features are fused to
predict 20 future motion steps. Waymo E2E V1.0.0 provides exactly this native
sample structure; it does not store 16 camera images inside each sample.

## Open in VS Code

Open the `driving_algorithm` folder, then select
`driving_algorithm/.venv/bin/python` with **Python: Select Interpreter**. The
scripts can also be opened and edited like ordinary Python files.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest tests -v
```

## Waymo Data Layout

Only the **End-to-End Driving Dataset V1.0.0** is used. The Perception and Motion
downloads are not required for this baseline.

```text
data/
├── DOWNLOAD_LOG.md
├── raw/waymo_e2e/                 # official TFRecord shards
├── converted/waymo_e2e/
│   ├── images/                    # current FRONT JPEG per sample
│   └── motion/                    # [16,8] history and [20,5] target NPZ
├── manifests/                     # model indexes
├── reports/                       # inspection and validation JSON
└── previews/waymo_e2e/            # presentation PNGs
```

Raw downloads require signing in at <https://waymo.com/open/download/> and
accepting the Waymo dataset terms. Credentials, cookies and signed URLs are
never stored in this project.

The prepared subset contains 7 training shards and 1 validation shard: 12,129
samples from 2,447 route prefixes. Raw plus converted data is about 29.2 GiB,
with no route or sample overlap between train and validation. Exact file sizes
and SHA-256 values are recorded in `data/DOWNLOAD_LOG.md`.

## Inspect and Convert

Inspect official TFRecord framing, camera fields and motion lengths:

```bash
.venv/bin/python -m driving_algorithm.waymo.inspect_records \
  data/raw/waymo_e2e/val_202504211843.tfrecord-00000-of-00093 \
  --report data/reports/waymo_first_shard.json
```

Convert one or more shards from the same official split:

```bash
.venv/bin/python -m driving_algorithm.waymo.convert_records \
  data/raw/waymo_e2e/val_*.tfrecord-* \
  --output-root data/converted/waymo_e2e \
  --manifest data/manifests/waymo_e2e_validation.csv \
  --split validation
```

Training shards use a separate `waymo_e2e_train.csv` manifest and
`--split train`. Pass every retained shard for that split whenever rebuilding
its manifest.

## Validate and Preview

```bash
.venv/bin/python -m driving_algorithm.waymo.validate_preparation \
  --manifest data/manifests/waymo_e2e_validation.csv \
  --data-root data/converted/waymo_e2e \
  --report data/reports/waymo_e2e_validation.json

.venv/bin/python -m driving_algorithm.waymo.preview_sequence \
  --manifest data/manifests/waymo_e2e_validation.csv \
  --data-root data/converted/waymo_e2e \
  --output data/previews/waymo_e2e/sample.png \
  --index 0
```

Validation checks duplicate IDs, route-level split leakage, missing files,
image decoding, motion shapes, finite values and the unified data contract.

## Algorithm Handoff

Group members can load a sample or batch without importing TensorFlow, Waymo
wheels or CARLA:

```python
from torch.utils.data import DataLoader
from driving_algorithm.data.waymo_dataset import WaymoE2EDataset

dataset = WaymoE2EDataset(
    "data/manifests/waymo_e2e_validation.csv",
    "data/converted/waymo_e2e",
)
batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))

print(batch["image"].shape)          # [8, 3, 224, 224] -> CNN
print(batch["state_history"].shape)  # [8, 16, 8] -> LSTM
print(batch["future_target"].shape)  # [8, 20, 5] -> training target
```

The later CARLA adapter must provide the same current image and 16-step ego
history. CARLA remains on the group's Ubuntu/NVIDIA machine; no CARLA server is
needed to prepare or train this dataset on the Mac.
