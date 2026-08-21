# Driving Prediction and Risk Interface Design

## 1. Purpose

Build the algorithm side of the honours project on the current Apple M1 Pro
Mac without requiring a local CARLA server. The deliverable trains a CNN-LSTM
future-motion predictor from real driving data, evaluates it offline, and
exposes a stable interface that the group can later connect to CARLA running on
an Ubuntu/NVIDIA computer.

The approved data strategy is:

1. retain comma2k19 as the real highway source;
2. add a storage-limited subset of the Waymo End-to-End Driving dataset for
   real urban driving;
3. let the CARLA workstream generate controlled roundabout, residential,
   intersection, minor-to-major-road, traffic-density and risk scenarios.

## 2. Scope

### In scope

- A source-independent sample contract for comma2k19, Waymo E2E and CARLA.
- A selected Waymo E2E download capped at 20-40 GB.
- Conversion of source-native data into model-ready sequence indexes.
- A PyTorch CNN-LSTM model for ego future-motion prediction.
- Training, checkpointing, evaluation and trajectory preview tools.
- ADE, FDE, position RMSE, speed error and heading error reports.
- A deterministic risk assessor that combines the predicted ego path with
  surrounding-actor states supplied by CARLA.
- A CARLA-facing Python contract and recorded-packet test fixture that work
  without importing or running CARLA locally.

### Out of scope

- Running the CARLA server on the Mac.
- Building CARLA maps or authoring the group's scenario catalogue.
- Training a camera-only object detector or multi-object tracker.
- Claiming that comma2k19 or Waymo supplies collision probability, severity or
  MTTC ground-truth labels.
- Downloading the complete Waymo dataset.

## 3. Delivery Phases

The project is implemented as three independently testable subprojects.

### Phase A: Local algorithm foundation

Create the unified types, a comma2k19 adapter, the CNN-LSTM model, training and
evaluation commands, synthetic unit tests, and a one-batch smoke test. This
phase must work using existing local files only.

### Phase B: Waymo urban data

Download one Waymo E2E shard first. Inspect sample IDs, camera availability and
trajectory fields before expanding the selection. Add a
one-time converter so the main PyTorch training environment does not depend on
Waymo's TensorFlow package or CARLA. Expand only while the raw and converted
data remain within the approved 20-40 GB budget.

### Phase C: CARLA integration contract

Implement and test an adapter against recorded dictionary/NPZ packets. The
CARLA group can then map live `carla.Image`, ego telemetry and actor snapshots
into the same contract on its Ubuntu machine.

## 4. Directory Layout

The existing `comma2k19/`, `comma2k19_data_preparation/` and
`carla_simulation/` directories remain intact.

```text
driving_algorithm/
├── README.md
├── requirements.txt
├── configs/
│   └── baseline.yaml
├── driving_algorithm/
│   ├── data/
│   │   ├── contracts.py
│   │   ├── comma2k19_adapter.py
│   │   ├── mixed_dataset.py
│   │   └── transforms.py
│   ├── waymo/
│   │   ├── inspect_records.py
│   │   └── convert_records.py
│   ├── models/
│   │   └── cnn_lstm.py
│   ├── risk/
│   │   └── assessor.py
│   ├── carla_contract.py
│   ├── train.py
│   ├── evaluate.py
│   └── preview.py
├── tests/
└── data/
    ├── raw/waymo_e2e/
    ├── converted/waymo_e2e/
    ├── manifests/
    ├── previews/
    └── reports/
```

Raw data, converted samples, checkpoints and generated reports are ignored by
Git. Source code, configuration, small test fixtures and documentation are
tracked.

## 5. Unified Data Contract

Each model-ready item is a dictionary with the following values:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `image` | float32 `[3, H, W]` | ImageNet-normalised current front RGB image |
| `state_history` | float32 `[T, 8]` | Normalised ego history |
| `future_target` | float32 `[K, 5]` | Unnormalised future motion |
| `history_mask` | bool `[T]` | Valid historical steps |
| `future_mask` | bool `[K]` | Valid prediction targets |
| `sample_id` | string | Stable source-qualified ID |
| `route_id` | string | Route/run ID used for splitting |
| `source` | string | `comma2k19`, `waymo_e2e` or `carla` |
| `split` | string | `train`, `validation` or `test` |
| `scene_type` | string | Known scene class or `unknown` |

The eight state features, in order, are local longitudinal position, local
lateral position, longitudinal velocity, lateral velocity, speed,
acceleration, relative heading and yaw rate. The five future values are local
longitudinal position, local lateral position, speed, acceleration and relative
heading. Source adapters derive these values deterministically from positions,
velocities and timestamps. A sample that cannot produce the complete feature
vectors is rejected and reported; missing values are never silently replaced
with plausible-looking labels.

Coordinates are ego-local at the final history step: positive x is forward and
positive y is left. Distances use metres, time uses seconds, velocity uses
metres per second, and angles use radians.

## 6. Time Semantics

The common model timeline is 4 Hz with 4 seconds of history and 5 seconds of
future prediction:

- `T = 16` historical steps;
- `K = 20` future steps;
- `dt = 0.25` seconds.

This matches the native Waymo E2E planning trajectory and gives every shape a
single physical meaning. comma2k19 is downsampled from 20 Hz when new unified
indexes are built; existing 30-history/20-future preparation outputs remain
available and are not overwritten. CARLA may run at a higher tick rate, but its
adapter supplies the predictor at 4 Hz.

The first real shard established the native Waymo E2E V1.0.0 sample semantics:
each record contains one current camera frame, a 16-step ego-state history and a
20-step future target. Its frame timestamp may be zero; `frame.context.name` is
the stable sample identifier. Before expanding the download, inspection must
prove that every selected record has these camera and motion fields. Missing
temporal images are never fabricated or repeated.

## 7. Dataset Rules

- Splits are route/run based. Overlapping windows from one route never cross
  train, validation and test boundaries.
- The baseline uses one front camera because comma2k19 has one front view.
- Waymo side and rear cameras are retained in raw files but excluded from the
  first model.
- Training batches use explicit source-aware sampling so the much larger
  comma2k19 index cannot eliminate urban Waymo samples.
- Scene labels are metadata for stratified reporting, not model targets in the
  baseline.
- Conversion is restartable and records source file, record timestamp, output
  checksum and conversion version in a manifest.

## 8. Model

The baseline model has four parts:

1. a pretrained ResNet-18 visual encoder applied to the current front image;
2. a small state encoder applied to each of the 16 ego-state steps;
3. an LSTM over the encoded state history;
4. a fusion layer combining current visual context and the final recurrent state;
5. a trajectory head that maps the fused representation to `[20, 5]`.

The visual backbone can be frozen from configuration for low-memory training.
The code selects Apple MPS when available and otherwise runs on CPU. It also
runs on CUDA without changing checkpoints or input fields.

Training uses masked Smooth L1 loss. Position is the primary term; speed,
acceleration and wrapped heading losses are auxiliary terms with explicit
configuration weights. Checkpoints include model state, optimizer state,
configuration, epoch, normalisation values and source manifest fingerprints.

## 9. Evaluation

Every evaluation report contains:

- ADE over valid x/y target points;
- FDE at the final valid x/y point;
- x/y position RMSE;
- speed MAE;
- wrapped heading MAE;
- metrics grouped by source and available scene type;
- sample count and invalid-sample count for every group.

A trajectory preview overlays history and prediction/ground truth in ego-local
coordinates and displays representative source frames. Tests use synthetic
tensors and a tiny recorded fixture; the test suite never requires full raw
datasets.

## 10. Risk Assessment and CARLA Boundary

The predictor estimates ego motion. CARLA supplies surrounding actors because
neither comma2k19 nor Waymo E2E supplies the final project-specific risk labels
in the required form.

The CARLA request contract contains:

- the latest RGB frame and its capture timestamp;
- ordered ego states in SI units;
- current actor ID, class, local x/y position, local x/y velocity and bounding
  box extent;
- scenario metadata such as map, scene type, density, weather and lighting.

The response contains:

- `future_trajectory` `[20, 5]`;
- `risk_score` in `[0, 1]`;
- `risk_level` as `low`, `medium` or `high`;
- `minimum_ttc_seconds` or null when no conflict exists;
- `minimum_separation_metres`;
- `conflict_actor_id` or null.

The first risk assessor uses constant-velocity actor extrapolation and compares
actor footprints against the predicted ego path at the same timestamps. Its
thresholds are configuration values and must be reported with results. This is
a deterministic integration baseline, not a learned collision probability.

## 11. Error Handling

- Missing raw data produces an actionable path and download instruction.
- Malformed rows report source file and record/sample ID.
- Duplicate sample IDs, insufficient history, non-finite motion values and
  inconsistent shapes are rejected during conversion.
- Waymo inspection reports compatibility before bulk conversion.
- Training refuses route leakage, mismatched normalisation dimensions and
  incompatible checkpoint/config combinations.
- CARLA contract validation rejects wrong units, missing timestamps and actor
  arrays with inconsistent lengths.

## 12. Acceptance Criteria

The approved algorithm work is complete when:

1. all unit and contract tests pass on the Mac without CARLA;
2. a comma2k19 batch has shapes `[B, 3, 224, 224]`, `[B, 16, 8]` and
   `[B, 20, 5]` under the unified configuration;
3. a one-batch forward/backward smoke test completes with finite loss;
4. at least one downloaded Waymo shard has a compatibility report and converted
   samples and previews;
5. mixed-source loading preserves route/run split isolation;
6. evaluation emits overall and source-stratified ADE/FDE reports;
7. the CARLA contract fixture produces a valid future trajectory and
   deterministic risk result without importing CARLA;
8. the README documents exact VS Code, training, evaluation and CARLA handoff
   commands.

## 13. References

- Waymo Open Dataset overview: <https://waymo.com/open/about/>
- Waymo E2E protocol definition:
  <https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto>
- Existing local data handoff: `comma2k19_data_preparation/ALGORITHM_HANDOFF.md`
- Existing remote CARLA environment: `carla_simulation/README.md`
