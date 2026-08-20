# Honours Project: Driving Risk Assessment

This repository contains the data-preparation and simulation code used for an
honours project on driving-risk assessment. The current pipeline prepares the
comma2k19 driving dataset for a CNN-LSTM future-motion model and provides a
CARLA environment for later collision-risk and severity experiments.

## Team guides

- [CARLA and dataset integration guide](CARLA_DATASET_GUIDE.md)
- [comma2k19 data preparation](comma2k19_data_preparation/README.md)
- [CARLA environment setup](carla_simulation/README.md)
- [Model handoff contract](comma2k19_data_preparation/ALGORITHM_HANDOFF.md)

## Repository contents

- `comma2k19_data_preparation/`: comma2k19 manifests, train/validation/test
  splits, preprocessing code, validation reports and representative previews.
- `carla_simulation/`: a CARLA 0.9.16 browser-based simulation environment.
- `dataset_scanner.py`: scans local trajectory CSV files for usable columns.
- `datasetup.py`: standardises selected trajectory datasets for CARLA work.
- `processed_data/`: small dataset scan reports that are safe to keep in Git.
- `DATA.md`: source-data download, layout and storage notes.

## comma2k19 preparation

The prepared index covers all ten comma2k19 chunks and contains 234,650
sequences. Each sample uses 30 historical frames (about 1.5 seconds) to predict
20 future frames (about 1 second). Routes, rather than individual frames, are
assigned to train, validation and test splits to avoid route leakage.

Create the Python environment from the project root:

```bash
python3 -m venv comma2k19_data_preparation/.venv
comma2k19_data_preparation/.venv/bin/python -m pip install \
  -r comma2k19_data_preparation/requirements.txt
```

After placing the raw data in `comma2k19/` as described in `DATA.md`, rebuild
and validate all prepared indices with:

```bash
comma2k19_data_preparation/.venv/bin/python \
  comma2k19_data_preparation/prepare_all_chunks.py --skip-existing
```

See `comma2k19_data_preparation/README.md` and
`comma2k19_data_preparation/ALGORITHM_HANDOFF.md` for the model-facing schema
and integration details.

## CARLA environment

The CARLA server requires an Ubuntu x86_64 host with an NVIDIA GPU. The current
Apple Silicon development machine can be used as the browser client. See
`carla_simulation/README.md` for setup and operating instructions.

## Data policy

The approximately 100 GB comma2k19 source dataset and other multi-gigabyte raw
datasets are intentionally excluded from Git history. This repository keeps
the reproducible code, derived CSV/JSON indices, validation reports and preview
assets. See `DATA.md` for the authoritative comma2k19 source and citation.
