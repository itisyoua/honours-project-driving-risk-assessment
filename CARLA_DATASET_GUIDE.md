# CARLA and Dataset Integration Guide

This guide is for team members responsible for the CARLA environment, traffic
scenarios, and risk labels. Start with the dataset boundaries below. The
repository can launch a CARLA environment with randomly generated traffic and
can read comma2k19 for future-motion model training, but it does not yet include
the adapter that replays CSV trajectories as CARLA NPCs.

## 1. Dataset responsibilities

| Data source | Purpose in this project | Directly importable into CARLA? |
| --- | --- | --- |
| comma2k19 | Train camera features, ego-state encoding, and future-motion prediction | No. Ego-vehicle data alone cannot reconstruct a complete traffic scene. |
| NGSIM / WholeVdata2 | Provide timestamps, positions, speeds, and trajectories for multiple vehicles | Not yet. Coordinate conversion and an NPC replay adapter are required. |
| Live CARLA simulation | Generate surrounding-vehicle states, collision events, MTTC, risk, and severity labels | Yes. This is the data source for the final risk stage. |

The recommended end-to-end workflow is:

```text
comma2k19 ----> CNN-LSTM ego future-motion model --┐
                                                   ├--> Risk fusion and evaluation
Trajectory data -> CARLA NPC traffic scenes -------┤
CARLA ----------> Collisions, relative states, labels ┘
```

Do not treat comma2k19 as a CARLA map. It records real-world road images,
sensor data, and ego motion; it cannot directly generate a 3D model of Highway
280 or all surrounding road users.

## 2. Runtime requirements

Run the CARLA 0.9.16 server on a machine with:

- Ubuntu 20.04 or 22.04 on x86_64;
- an NVIDIA GPU with at least 8 GB VRAM recommended;
- an NVIDIA driver, Docker, the Docker Compose plugin, and NVIDIA Container Toolkit;
- Python 3.10, 3.11, or 3.12;
- at least 20 GB of free disk space, excluding source datasets.

An Apple Silicon Mac cannot run the current CARLA server image natively, but it
can access the browser dashboard hosted by the Ubuntu machine.

## 3. Clone the project and start a basic CARLA scene

```bash
git clone https://github.com/itisyoua/honours-project-driving-risk-assessment.git
cd honours-project-driving-risk-assessment/carla_simulation

python3 doctor.py
./launch.sh --no-open
```

On the Ubuntu host, open:

```text
http://127.0.0.1:8080
```

From another computer on the same network, open:

```text
http://<UBUNTU_HOST_IP>:8080
```

The default configuration loads `Town03`, spawns one ego vehicle and 20 random
NPCs, and attaches an RGB camera and collision sensor. Edit
`carla_simulation/.env` to change the scene:

```dotenv
CARLA_TOWN=Town05
CARLA_NPC_VEHICLES=30
CARLA_SEED=42
```

Stop the environment with:

```bash
cd carla_simulation
./stop.sh
```

## 4. Download and validate comma2k19

The GitHub repository does not contain the approximately 100 GB comma2k19 raw
dataset. Download it from the official source listed in [`DATA.md`](DATA.md),
then extract it into this layout:

```text
honours-project-driving-risk-assessment/
├── comma2k19/
│   ├── Chunk_1/
│   ├── Chunk_2/
│   ├── ...
│   └── Chunk_10/
└── comma2k19_data_preparation/
```

Install the Python dependencies:

```bash
cd honours-project-driving-risk-assessment
python3 -m venv comma2k19_data_preparation/.venv
comma2k19_data_preparation/.venv/bin/python -m pip install \
  -r comma2k19_data_preparation/requirements.txt
```

Read one validation sample:

```bash
comma2k19_data_preparation/.venv/bin/python \
  -m comma2k19_data_preparation.comma2k19_dataset
```

A successful run prints a sample `sequence_id` and these tensor shapes:

```text
frames:        (30, 3, 224, 224)
state_history: (30, 8)
future_target: (20, 5)
```

Model code must use the provided route-level `train`, `validation`, and `test`
splits. Do not randomly split individual sliding windows because adjacent
overlapping sequences would cause data leakage.

## 5. Prepare NGSIM / WholeVdata2 trajectories

These trajectory datasets are more suitable than comma2k19 for controlling
CARLA NPCs. Their raw files are not stored in Git. Obtain them separately under
their respective licences and place them in the project-level `raw_data/`
directory.

The current script reads only these two filenames:

```text
Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
WholeVdata2.csv
```

Run:

```bash
cd honours-project-driving-risk-assessment
python3 -m venv .venv
.venv/bin/python -m pip install pandas numpy
.venv/bin/python datasetup.py
```

The main output is:

```text
processed_data/usable_carla_ready_trajectories.csv
```

It contains:

```text
source_file, vehicle_id, frame_id, time, x, y,
vx, vy, speed, acceleration, heading_rad, heading_deg,
lane_id, vehicle_type, nearest_vehicle_id, nearest_distance
```

The `carla_ready` filename means the columns have been standardised; it does not
mean the coordinates can already be passed to CARLA. Confirm the units in each
source dataset. Some NGSIM position and speed fields may use imperial units and
must be converted to metres, metres per second, and seconds before replay.

## 6. Implement the trajectory-to-CARLA NPC adapter

Add `carla_simulation/trajectory_replay.py` and develop it in the following
order.

### 6.1 Read and select a scene

1. Read `usable_carla_ready_trajectories.csv`.
2. Select a continuous time window.
3. Identify the ego vehicle and surrounding vehicles in that window.
4. Resample every vehicle timestamp to the CARLA tick rate.

The current CARLA environment uses a fixed `0.05 s` step (20 Hz). If the source
trajectory is 10 Hz, interpolate it to 20 Hz or change CARLA's
`fixed_delta_seconds` consistently.

### 6.2 Convert coordinates

Dataset coordinates cannot be used directly as CARLA world coordinates. Define
one scene-level transform:

```text
CARLA_xy = rotation(scale * dataset_xy) + translation
```

Validate at least:

- whether distance units are metres;
- the x/y axis directions;
- the heading zero point and clockwise/counter-clockwise convention;
- whether dataset lane widths match the CARLA map;
- whether every spawn point lies close to a drivable lane.

For the first implementation, align trajectories with a straight section of
`Town04` or `Town05` instead of attempting to reconstruct Highway 280.

### 6.3 Spawn and control vehicles

1. Create one CARLA vehicle actor for each `vehicle_id`.
2. Select a similar CARLA blueprint from the vehicle type field.
3. Update vehicle location and rotation on every tick.
4. Use `set_transform` for deterministic replay in the first version.
5. Switch to a PID/controller producing throttle, brake, and steering when
   realistic dynamics are required.
6. Destroy every actor at the end of the scene to avoid leftover vehicles on
   repeated runs.

### 6.4 Record risk inputs and labels

Record at least the following on every tick:

- ego and surrounding-vehicle position, velocity, acceleration, and heading;
- vehicle bounding-box dimensions and relative distance;
- nearest leading vehicle, relative speed, and TTC/MTTC;
- scenario ID, weather, map, and random seed;
- collision time, collision actor, and collision impulse;
- model warning time and warning lead time.

Collision-probability and severity labels must be generated from CARLA scene
data. Do not treat comma2k19 `future_target` values as collision labels.

## 7. Connect the future-motion model

After integrating the model with CARLA, maintain a rolling buffer containing
the latest 30 frames. The CARLA camera and ego state form:

```text
frames        -> [30, 3, 224, 224]
state_history -> [30, 8]
```

The model outputs:

```text
future_target -> [20, 5]
```

Pass the predicted ego trajectory and CARLA NPC trajectories to the risk module
to calculate future-path intersections, TTC/MTTC, collision probability, and
severity.

## 8. Recommended team responsibilities

| Owner | Main work | Deliverables |
| --- | --- | --- |
| Data/model | comma2k19 loading, CNN-LSTM, future-trajectory evaluation | Model weights, inference interface, ADE/FDE |
| CARLA scenes | Coordinate conversion, NPC replay, sensor recording | `trajectory_replay.py`, scene configuration |
| Risk module | TTC/MTTC, collision probability, severity | Risk calculation interface and evaluation report |
| Integration testing | Fixed random seeds, repeatable experiments, result aggregation | Reproducible commands and test records |

## 9. Definition of done

Use this checklist to accept the CARLA dataset integration:

- [ ] `launch.sh` starts CARLA successfully on the Ubuntu host.
- [ ] Other team members can view the browser dashboard.
- [ ] The comma2k19 single-sample test passes.
- [ ] Trajectory units and sampling frequency have been confirmed.
- [ ] At least one CSV scene can be replayed in CARLA.
- [ ] The same random seed reproduces the same scene.
- [ ] NPCs do not spawn off-road or overlap at scene start.
- [ ] Ego and NPC state is saved on every tick.
- [ ] Collision events, TTC/MTTC, and warning lead time can be calculated.
- [ ] Every experiment records its map, weather, scenario ID, and commit hash.

## 10. Troubleshooting

### The CARLA container does not start

Run:

```bash
cd carla_simulation
python3 doctor.py
docker compose logs carla
```

Confirm that `nvidia-smi` works and that Docker can access the GPU.

### The browser cannot connect to port 8080

Confirm that `.env` contains `WEB_HOST=0.0.0.0`, check the Ubuntu firewall, and
open the port only on a trusted local network. The current browser dashboard
does not provide authentication and must not be exposed directly to the public
internet.

### `Comma2k19Dataset` cannot find a video

Confirm that `comma2k19/` and `comma2k19_data_preparation/` are in the same
project root and that the dataset directories are named `Chunk_1` through
`Chunk_10`.

### Vehicle trajectories do not align with the road

Do not fix this by repeatedly adjusting individual vehicle positions. Validate
the units, origin, rotation, axis directions, and selected map section, then
apply one consistent scene-level transform to every vehicle.
