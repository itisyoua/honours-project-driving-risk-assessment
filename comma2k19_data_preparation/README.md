# comma2k19 Data Preparation

This folder is the data handoff for the project's CNN-LSTM future-motion model.
It prepares comma2k19 without moving or modifying the raw videos.

## Current dataset configuration

- Source data: Chunk_1 through Chunk_10
- Historical input: 30 frames (approximately 1.5 seconds)
- Future target: 20 frames (approximately 1.0 second)
- Sliding-window stride: 10 frames
- Image tensor size: `30 x 3 x 224 x 224`
- State tensor size: `30 x 8`
- Future target size: `20 x 5`
- Split rule: entire routes are assigned to train, validation or test

## Files

- `comma2k19_manifest.py`: checks raw data, creates future-motion windows and route splits.
- `comma2k19_utils.py`: shared video decoding and local-coordinate motion features.
- `comma2k19_dataset.py`: model-ready PyTorch Dataset.
- `combine_chunk_results.py`: builds one training index from all prepared chunks.
- `prepare_all_chunks.py`: one command that prepares, combines and validates every extracted chunk.
- `preview_sequence.py`: creates a PNG overview and animated GIF for one sequence.
- `generate_preview_catalog.py`: creates nine representative previews per chunk,
  stratified by split and low/medium/high speed.
- `validate_preparation.py`: checks split leakage, paths, numeric values and tensor shapes.
- `requirements.txt`: reproducible Python dependencies.
- `chunk_1_results/` and `chunk_2_results/`: per-chunk outputs and validation.
- `combined_results/`: the default algorithm input covering both chunks.

## Run in VSCode

The workspace interpreter points to `comma2k19_data_preparation/.venv/bin/python`.
Open a script and choose **Run Python File**.

The complete one-command workflow from the project root is:

```bash
comma2k19_data_preparation/.venv/bin/python comma2k19_data_preparation/prepare_all_chunks.py
```

Use `--skip-existing` to keep existing per-chunk results and only process newly
extracted chunks before rebuilding the combined index.

To recreate the environment on another computer:

```bash
python3 -m venv comma2k19_data_preparation/.venv
comma2k19_data_preparation/.venv/bin/python -m pip install -r comma2k19_data_preparation/requirements.txt
```

## PyTorch handoff

```python
from torch.utils.data import DataLoader
from comma2k19_data_preparation import Comma2k19Dataset

train_dataset = Comma2k19Dataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

batch = next(iter(train_loader))
print(batch["frames"].shape)         # [B, 30, 3, 224, 224]
print(batch["state_history"].shape)  # [B, 30, 8]
print(batch["future_target"].shape)  # [B, 20, 5]
```

The default Dataset reads `combined_results`, containing 234,650 sequences from
all ten chunks. A single chunk can be selected by passing its sequence-index CSV.

The eight state features are local longitudinal/lateral position, local
longitudinal/lateral velocity, speed, acceleration, relative heading and yaw
rate. The five prediction targets are future local longitudinal/lateral
position, speed, acceleration and relative heading.

Local coordinates are centred at the final historical frame. This avoids
feeding large absolute ECEF/GPS coordinates into the model.

## Important scope boundary

comma2k19 provides real road images and ego-vehicle future motion targets. It
does not provide final collision probability, severity or MTTC labels. Those
values must be generated and validated with surrounding-actor states in CARLA.

Raw HEVC files do not contain seek timestamps, so decoding samples late in a
one-minute segment is slower than decoding early samples. The interface is
correct for model integration; performance caching can be added after the
group confirms the final CNN input resolution and sampling rate.

## Preview catalogue

Each `chunk_N_results/previews/catalog` folder contains nine representative
PNG/GIF pairs, a `preview_catalog.csv`, and an `overview.png`. Rows correspond
to train/validation/test and columns correspond to low/medium/high speed. The
file `preview_overview_all_chunks.png` provides one combined overview.
