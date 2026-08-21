# Waymo E2E Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download a storage-limited Waymo End-to-End Driving subset and convert temporally compatible front-camera records into validated 4 Hz CNN-LSTM sequence manifests matching the project's unified data contract.

**Architecture:** A Mac-compatible, TensorFlow-free converter reads TFRecord framing and Waymo protobuf messages, inspects one shard before bulk work, then stores each front JPEG once per run plus compressed motion arrays and CSV indexes. Existing comma2k19 files remain unchanged; shared motion utilities make both sources expose 16 history steps, 20 future steps, eight history features and five future features.

**Tech Stack:** Python 3.9+, PyTorch 2.x, NumPy, Pillow, protobuf, grpcio-tools, google-crc32c, pytest, CSV/JSON/NPZ.

**Spec:** `docs/superpowers/specs/2026-08-21-driving-algorithm-design.md`

## Global Constraints

- Development machine is Apple M1 Pro on macOS arm64; CARLA and Linux-only Waymo wheels are not runtime dependencies.
- Waymo raw plus converted data must remain within 40 GB, with an initial one-shard inspection before expansion.
- Common timeline is exactly 4 Hz: 16 history steps and 20 future steps.
- Baseline camera is Waymo `FRONT` only; other camera bytes are not extracted.
- Coordinates are ego-local with positive x forward, positive y left, SI units and radians.
- Route/run groups never cross train, validation and test splits.
- Existing `comma2k19/` and `comma2k19_data_preparation/` outputs are never moved, overwritten or deleted.
- Generated data, raw TFRecords and generated previews are excluded from Git.
- Waymo protobuf sources are pinned to official revision `99a4cb3ff07e2fe06c2ce73da001f850f628e45a`.

---

### Task 1: Create the Data Package and Contract

**Files:**
- Create: `driving_algorithm/requirements.txt`
- Create: `driving_algorithm/README.md`
- Create: `driving_algorithm/driving_algorithm/__init__.py`
- Create: `driving_algorithm/driving_algorithm/data/__init__.py`
- Create: `driving_algorithm/driving_algorithm/data/contracts.py`
- Create: `driving_algorithm/tests/test_contracts.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `SequenceContract.validate(sample: Mapping[str, object]) -> None`
- Produces: constants `HISTORY_STEPS = 16`, `FUTURE_STEPS = 20`, `STATE_DIM = 8`, `TARGET_DIM = 5`, `SAMPLE_HZ = 4.0`
- Produces: `make_sample_id(source: str, route_id: str, timestamp_micros: int) -> str`

- [ ] **Step 1: Write the failing contract tests**

```python
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


def test_sample_id_is_source_qualified():
    assert make_sample_id("waymo_e2e", "run-1", 1_000_000) == "waymo_e2e:run-1:1000000"
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_contracts.py -v`

Expected: FAIL because `driving_algorithm.data.contracts` does not exist.

- [ ] **Step 3: Implement the constants and strict validator**

Implement `SequenceContract.validate` to check required keys, exact array shapes, dtypes convertible to NumPy, finite motion values, boolean mask shapes, allowed split names and non-empty IDs. Do not accept alternate dimensions.

- [ ] **Step 4: Add dependencies and ignore generated artifacts**

`requirements.txt` contains exact compatible dependency families:

```text
numpy>=2.0,<3
pillow>=11,<13
matplotlib>=3.9,<4
torch>=2.0,<3
torchvision>=0.20,<1
protobuf>=5,<7
grpcio-tools>=1.70,<2
google-crc32c>=1.6,<2
pytest>=8,<10
```

Add these ignore rules:

```gitignore
driving_algorithm/.venv/
driving_algorithm/data/raw/
driving_algorithm/data/converted/
driving_algorithm/data/previews/
driving_algorithm/data/reports/
driving_algorithm/data/manifests/*.csv
driving_algorithm/data/manifests/*.json
driving_algorithm/checkpoints/
```

- [ ] **Step 5: Run the contract tests**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the contract**

```bash
git add .gitignore driving_algorithm
git commit -m "feat: add unified driving sequence contract"
```

---

### Task 2: Add Motion Feature Derivation and Timeline Validation

**Files:**
- Create: `driving_algorithm/driving_algorithm/data/motion.py`
- Create: `driving_algorithm/tests/test_motion.py`

**Interfaces:**
- Consumes: timing constants from `driving_algorithm.data.contracts`
- Produces: `derive_state_history(pos_x, pos_y, vel_x, vel_y, accel_x, accel_y, dt) -> np.ndarray`
- Produces: `derive_future_target(pos_x, pos_y, initial_velocity_xy, dt) -> np.ndarray`
- Produces: `validate_timestamps(timestamps_micros, expected_dt_micros=250_000, tolerance_micros=25_000) -> None`

- [ ] **Step 1: Write failing tests for straight and turning motion**

```python
import numpy as np
import pytest

from driving_algorithm.data.motion import (
    derive_future_target,
    derive_state_history,
    validate_timestamps,
)


def test_straight_history_has_expected_speed_and_heading():
    x = np.arange(16, dtype=np.float32) * 2.5
    y = np.zeros(16, dtype=np.float32)
    vx = np.full(16, 10.0, dtype=np.float32)
    vy = np.zeros(16, dtype=np.float32)
    ax = np.zeros(16, dtype=np.float32)
    ay = np.zeros(16, dtype=np.float32)
    state = derive_state_history(x, y, vx, vy, ax, ay, dt=0.25)
    assert state.shape == (16, 8)
    np.testing.assert_allclose(state[:, 4], 10.0)
    np.testing.assert_allclose(state[:, 6], 0.0)


def test_future_target_uses_local_origin_and_five_features():
    x = np.arange(1, 21, dtype=np.float32) * 2.5
    y = np.zeros(20, dtype=np.float32)
    target = derive_future_target(x, y, np.array([10.0, 0.0]), dt=0.25)
    assert target.shape == (20, 5)
    np.testing.assert_allclose(target[-1, :2], [50.0, 0.0])


def test_timestamp_gap_is_rejected():
    timestamps = np.arange(16, dtype=np.int64) * 250_000
    timestamps[8:] += 100_000
    with pytest.raises(ValueError, match="timestamp"):
        validate_timestamps(timestamps)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_motion.py -v`

Expected: FAIL because `motion.py` does not exist.

- [ ] **Step 3: Implement deterministic feature derivation**

Derive speed with vector magnitude, acceleration with longitudinal projection, heading with `atan2(vy, vx)`, relative heading against the final history heading, and yaw rate with unwrapped finite differences. Future speed, acceleration and heading are finite differences of future x/y beginning from the supplied current velocity. Return float32 arrays and reject non-finite or wrong-length inputs.

- [ ] **Step 4: Run motion and contract tests**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_motion.py tests/test_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit motion utilities**

```bash
git add driving_algorithm/driving_algorithm/data/motion.py driving_algorithm/tests/test_motion.py
git commit -m "feat: derive unified ego motion features"
```

---

### Task 3: Bootstrap Official Waymo Protobuf Definitions

**Files:**
- Create: `driving_algorithm/scripts/bootstrap_waymo_protos.py`
- Create: `driving_algorithm/waymo_open_dataset/__init__.py`
- Create: `driving_algorithm/waymo_open_dataset/protos/__init__.py`
- Create after generation: `driving_algorithm/waymo_open_dataset/**/*_pb2.py`
- Create: `driving_algorithm/THIRD_PARTY_NOTICES.md`
- Create: `driving_algorithm/tests/test_waymo_proto.py`

**Interfaces:**
- Produces: importable `waymo_open_dataset.protos.end_to_end_driving_data_pb2.E2EDFrame`
- Produces: `bootstrap_waymo_protos.py --output-root PATH`

- [ ] **Step 1: Write the failing protobuf smoke test**

```python
from waymo_open_dataset.protos import end_to_end_driving_data_pb2


def test_e2e_proto_round_trip():
    frame = end_to_end_driving_data_pb2.E2EDFrame()
    frame.frame.context.name = "run-1"
    frame.frame.timestamp_micros = 1_000_000
    frame.past_states.pos_x.extend([0.0] * 16)
    parsed = end_to_end_driving_data_pb2.E2EDFrame.FromString(
        frame.SerializeToString()
    )
    assert parsed.frame.context.name == "run-1"
    assert len(parsed.past_states.pos_x) == 16
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_waymo_proto.py -v`

Expected: FAIL because generated Waymo protobuf modules do not exist.

- [ ] **Step 3: Implement the pinned bootstrap script**

The script downloads `.proto` files only from:

```text
https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/
```

It recursively resolves imports beginning at `waymo_open_dataset/protos/end_to_end_driving_data.proto`, stores sources in a temporary directory, invokes `python -m grpc_tools.protoc`, and writes generated modules under `driving_algorithm/waymo_open_dataset`. It verifies SHA-pinned URLs, refuses non-HTTPS redirects, and exits non-zero on any missing import.

- [ ] **Step 4: Generate the protobuf modules**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python scripts/bootstrap_waymo_protos.py --output-root .`

Expected: generated `dataset_pb2.py`, `label_pb2.py`, `end_to_end_driving_data_pb2.py` and their imported map/vector modules exist.

- [ ] **Step 5: Document third-party provenance**

Record the repository, exact revision, Apache-2.0 license URL, generated file list and generation command in `THIRD_PARTY_NOTICES.md`.

- [ ] **Step 6: Run the protobuf smoke test**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_waymo_proto.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the bootstrap and generated modules**

```bash
git add driving_algorithm/scripts driving_algorithm/waymo_open_dataset driving_algorithm/THIRD_PARTY_NOTICES.md driving_algorithm/tests/test_waymo_proto.py
git commit -m "feat: add Mac-compatible Waymo protobuf support"
```

---

### Task 4: Implement TensorFlow-Free TFRecord Reading and Shard Inspection

**Files:**
- Create: `driving_algorithm/driving_algorithm/waymo/__init__.py`
- Create: `driving_algorithm/driving_algorithm/waymo/tfrecord.py`
- Create: `driving_algorithm/driving_algorithm/waymo/records.py`
- Create: `driving_algorithm/driving_algorithm/waymo/inspect_records.py`
- Create: `driving_algorithm/tests/test_tfrecord.py`
- Create: `driving_algorithm/tests/test_waymo_records.py`

**Interfaces:**
- Consumes: generated `E2EDFrame` protobuf
- Produces: `iter_tfrecord(path: Path, verify_crc: bool = True) -> Iterator[bytes]`
- Produces: `parse_e2e_record(payload: bytes) -> E2EDFrame`
- Produces: `inspect_shards(paths: Sequence[Path], report_path: Path) -> dict`
- Produces CLI: `python -m driving_algorithm.waymo.inspect_records TFRECORD... --report REPORT.json`

- [ ] **Step 1: Write failing TFRecord tests**

Create a helper in the test that writes standard TFRecord framing: uint64 little-endian length, masked CRC32C of the length bytes, payload, and masked CRC32C of the payload. Test that two payloads round-trip and that a changed payload raises `ValueError` containing the record number and `CRC`.

- [ ] **Step 2: Write failing Waymo record inspection tests**

Build six serialized `E2EDFrame` messages with one run ID, 250,000-microsecond timestamp increments, a `FRONT` JPEG created with Pillow, 16 past states and 20 future positions. Assert that inspection reports:

```python
assert report["records"] == 6
assert report["runs"] == 1
assert report["front_camera_records"] == 6
assert report["past_length_counts"] == {"16": 6}
assert report["future_length_counts"] == {"20": 6}
assert report["timestamp_gap_micros"]["median"] == 250_000
```

- [ ] **Step 3: Run both test files and verify failure**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_tfrecord.py tests/test_waymo_records.py -v`

Expected: FAIL because reader and inspector modules do not exist.

- [ ] **Step 4: Implement framing, parsing and inspection**

The report must include file sizes, record count, run IDs, timestamp range, timestamp gaps per run, camera-name counts, image dimensions, past/future field length distributions, driving-intent counts, malformed record count and a boolean `compatible_with_16_frame_history`. Compatibility is true only when at least one run has 16 monotonically ordered front frames at approximately 4 Hz and every selected target record has 16 past and 20 future positions.

- [ ] **Step 5: Run TFRecord and inspector tests**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_tfrecord.py tests/test_waymo_records.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the reader and inspector**

```bash
git add driving_algorithm/driving_algorithm/waymo driving_algorithm/tests/test_tfrecord.py driving_algorithm/tests/test_waymo_records.py
git commit -m "feat: inspect Waymo E2E TFRecord shards"
```

---

### Task 5: Download and Inspect the First Authenticated Waymo Shard

**Files:**
- Create at runtime: `driving_algorithm/data/raw/waymo_e2e/`
- Create at runtime: `driving_algorithm/data/reports/waymo_first_shard.json`
- Create: `driving_algorithm/data/DOWNLOAD_LOG.md`

**Interfaces:**
- Consumes: one official E2E `training.tfrecord*` or `validation.tfrecord*` file
- Produces: the compatibility report required before conversion

- [ ] **Step 1: Sign in and accept the dataset terms**

Open <https://waymo.com/open/download/> in the in-app browser. The user handles Google authentication or any consent screen containing private credentials. Select the End-to-End Driving dataset in E2ED TFRecord format.

- [ ] **Step 2: Download exactly one shard**

Download one validation shard when available because the official validation release includes additional ratings/scenario metadata; otherwise download the first training shard. Move the completed file into `driving_algorithm/data/raw/waymo_e2e/` without renaming it.

- [ ] **Step 3: Record provenance and storage**

`DOWNLOAD_LOG.md` records download date, official dataset/version, split, original filename, byte size, SHA-256, license URL and the fact that access was authenticated. It contains no account email, token, cookie or signed URL.

- [ ] **Step 4: Run first-shard inspection**

Run the inspector against the sole file in `data/raw/waymo_e2e/` and write `data/reports/waymo_first_shard.json`. The command discovers and prints the actual downloaded filename rather than relying on a guessed shard name.

Expected: zero malformed records and `compatible_with_16_frame_history: true`.

- [ ] **Step 5: Apply the compatibility gate**

If compatibility is false, stop bulk download and write the concrete failed condition into `DOWNLOAD_LOG.md`. If true, proceed to Task 6. This gate cannot be bypassed by repeating frames or interpolating images.

- [ ] **Step 6: Commit provenance only**

```bash
git add driving_algorithm/data/DOWNLOAD_LOG.md
git commit -m "docs: record Waymo E2E source provenance"
```

---

### Task 6: Convert Compatible Waymo Runs into Indexed Sequences

**Files:**
- Create: `driving_algorithm/driving_algorithm/waymo/convert_records.py`
- Create: `driving_algorithm/driving_algorithm/data/waymo_dataset.py`
- Create: `driving_algorithm/tests/test_waymo_conversion.py`
- Create: `driving_algorithm/tests/test_waymo_dataset.py`
- Create at runtime: `driving_algorithm/data/converted/waymo_e2e/`
- Create at runtime: `driving_algorithm/data/manifests/waymo_e2e_*.csv`
- Create at runtime: `driving_algorithm/data/reports/waymo_e2e_summary.json`

**Interfaces:**
- Consumes: ordered E2ED records from Task 4
- Consumes: motion derivation from Task 2
- Produces: `convert_shards(paths, output_root, split, stride=4) -> ConversionSummary`
- Produces: `WaymoE2EDataset(index_csv, image_size=(224, 224), normalise_images=True)`
- Produces each sample conforming to `SequenceContract`

- [ ] **Step 1: Write a failing conversion test**

Create 20 synthetic sequential E2ED records for one run. Convert with a four-record stride and assert:

```python
assert summary.records == 20
assert summary.runs == 1
assert summary.sequences == 2
assert len(list((output / "runs" / "run-1" / "front").glob("*.jpg"))) == 20
assert (output / "runs" / "run-1" / "motion.npz").exists()
```

The expected two samples end at record indexes 15 and 19. JPEG count proves that overlapping windows do not duplicate image storage.

- [ ] **Step 2: Write a failing dataset test**

Load the generated CSV and assert one item has exact shapes `[16, 3, 224, 224]`, `[16, 8]` and `[20, 5]`, all finite values, `source == "waymo_e2e"`, and a route ID matching the synthetic run.

- [ ] **Step 3: Run tests and verify failure**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_waymo_conversion.py tests/test_waymo_dataset.py -v`

Expected: FAIL because converter and dataset modules do not exist.

- [ ] **Step 4: Implement storage-efficient conversion**

For each run, write each front JPEG once using its timestamp as the filename. Write `frame_index.csv` with frame index, timestamp and relative JPEG path. Write one `motion.npz` containing record timestamps, state histories, future targets, masks and intent values. Write manifest rows referencing run directory, history start/end indexes and target record index. Use temporary files followed by atomic rename so interrupted conversion is restartable.

- [ ] **Step 5: Preserve official splits and derive scene metadata**

Use the downloaded Waymo split as the manifest split. Map intent values to `unknown`, `go_straight`, `go_left` and `go_right`. Set `traffic_density` to `unknown`; do not infer it from image appearance without labels.

- [ ] **Step 6: Implement loading and ImageNet normalisation**

The dataset resolves only paths under its configured data root, loads the 16 indexed JPEGs with Pillow, converts to RGB, resizes to 224x224, applies existing comma2k19 ImageNet mean/std values, loads the indexed motion arrays and invokes `SequenceContract.validate` before returning.

- [ ] **Step 7: Run conversion and dataset tests**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_waymo_conversion.py tests/test_waymo_dataset.py -v`

Expected: all tests PASS.

- [ ] **Step 8: Convert the inspected real shard**

Run the converter with the actual sole shard discovered in Task 5, its official split, output root `data/converted/waymo_e2e`, manifest directory `data/manifests`, and stride `4`. Validate that total raw plus converted disk usage remains below 40 GB before downloading another shard.

- [ ] **Step 9: Commit conversion code and tests**

```bash
git add driving_algorithm/driving_algorithm/waymo/convert_records.py driving_algorithm/driving_algorithm/data/waymo_dataset.py driving_algorithm/tests/test_waymo_conversion.py driving_algorithm/tests/test_waymo_dataset.py
git commit -m "feat: convert Waymo E2E records into model sequences"
```

---

### Task 7: Add Validation, Summary and Preview Parity with comma2k19

**Files:**
- Create: `driving_algorithm/driving_algorithm/waymo/validate_preparation.py`
- Create: `driving_algorithm/driving_algorithm/waymo/preview_sequence.py`
- Create: `driving_algorithm/tests/test_waymo_validation.py`
- Modify: `driving_algorithm/README.md`
- Create at runtime: `driving_algorithm/data/previews/waymo_e2e/sequence_preview.png`
- Create at runtime: `driving_algorithm/data/reports/waymo_e2e_validation.json`

**Interfaces:**
- Consumes: Waymo manifest and converted run folders
- Produces: `validate_preparation(index_csv, data_root) -> dict`
- Produces CLI validation and PNG preview commands

- [ ] **Step 1: Write failing validation tests**

Assert that a valid synthetic conversion reports no errors. Then edit a copy of the manifest so one route appears in train and validation and assert that the validator reports `route_split_leakage`. Delete one JPEG and assert that it reports the exact missing relative path.

- [ ] **Step 2: Run validation tests and verify failure**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests/test_waymo_validation.py -v`

Expected: FAIL because validator does not exist.

- [ ] **Step 3: Implement validation and summary output**

Check manifest columns, unique sample IDs, split isolation, monotonic frame timestamps, exact temporal lengths, file existence, image decode, motion array shapes, finite values and data-contract validity. Report sequence/run counts, split counts, intent counts, speed distribution, total bytes, rejected samples and rejection reasons.

- [ ] **Step 4: Implement preview output**

Create a PNG matching the established comma2k19 concept: six evenly selected history frames in a 2x3 grid, followed by an ego-local plot with blue history, red future target, origin marker, sample/run/split/source labels and final forward/lateral displacement.

- [ ] **Step 5: Run validation tests and the full suite**

Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Validate and preview the real shard conversion**

Run validation against the generated real Waymo index and create at least one preview. Expected: zero route leakage, zero missing files, zero non-finite arrays, exact `[16, 8]`/`[20, 5]` motion shapes and a readable preview PNG.

- [ ] **Step 7: Expand to a useful train/validation subset**

After the real validation shard passes conversion and validation, download and
convert training shards one at a time. Keep at least one validation shard and
one training shard. Before each additional download, add its displayed size to
current raw plus converted usage; stop before the projected total would exceed
40 GB. Do not download the official test split because it has no future target
labels for supervised preprocessing. Re-run validation after every shard and
stop immediately on a new schema, timestamp or shape error.

- [ ] **Step 8: Document VS Code and handoff commands**

README instructions cover environment creation, authenticated download boundary, first-shard inspection, conversion, validation, preview generation, sample loading, storage-budget check and the fact that CARLA is not required.

- [ ] **Step 9: Commit validation and documentation**

```bash
git add driving_algorithm/driving_algorithm/waymo/validate_preparation.py driving_algorithm/driving_algorithm/waymo/preview_sequence.py driving_algorithm/tests/test_waymo_validation.py driving_algorithm/README.md
git commit -m "feat: validate and preview Waymo preparation"
```

---

## Final Verification

- [ ] Run: `cd driving_algorithm && ../comma2k19_data_preparation/.venv/bin/python -m pytest tests -v`
- [ ] Run: `git diff --check`
- [ ] Confirm raw plus converted Waymo data is below 40 GB with `du -sh driving_algorithm/data/raw/waymo_e2e driving_algorithm/data/converted/waymo_e2e`.
- [ ] Confirm the real validation report has zero errors.
- [ ] Open the generated PNG and visually check image order, labels and trajectory orientation.
- [ ] Load one real item and print exact tensor shapes, IDs, split, source and scene type.
