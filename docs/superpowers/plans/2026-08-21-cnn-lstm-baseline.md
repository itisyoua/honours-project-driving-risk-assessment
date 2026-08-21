# CNN-LSTM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the first trainable CNN-LSTM ego-trajectory predictor on the unified Waymo sample contract.

**Architecture:** ResNet-18 encodes the current front image while an MLP and LSTM encode the 16-step ego-state history. A fusion head predicts 20 future steps with five values per step; masked Smooth L1 losses and trajectory metrics support training and evaluation on Mac MPS, CUDA or CPU.

**Tech Stack:** Python 3.9+, PyTorch 2.x, torchvision, NumPy, pytest, JSON.

**Spec:** `docs/superpowers/specs/2026-08-21-driving-algorithm-design.md`

## Global Constraints

- Input shapes are image `[B,3,224,224]` and state history `[B,16,8]`.
- Output shape is `[B,20,5]` in ego-local SI units and radians.
- Tests never download pretrained weights or load the full raw dataset.
- The model runs unchanged on CPU, Apple MPS and CUDA.
- Checkpoints include model configuration, optimizer state, epoch and manifest fingerprint.

---

### Task 1: Motion Statistics and Device Selection

**Files:**
- Create: `driving_algorithm/driving_algorithm/data/statistics.py`
- Create: `driving_algorithm/driving_algorithm/runtime.py`
- Create: `driving_algorithm/tests/test_statistics.py`

**Interfaces:**
- Produces: `compute_state_statistics(manifest_path, data_root) -> StateStatistics`
- Produces: `select_device(preference="auto") -> torch.device`

- [ ] Write tests for exact mean/std over two small NPZ motion files and explicit CPU selection.
- [ ] Run the tests and verify imports fail.
- [ ] Implement masked float64 accumulation, minimum standard deviation clamping and device priority `mps`, `cuda`, `cpu`.
- [ ] Run tests and commit.

---

### Task 2: CNN-LSTM Predictor

**Files:**
- Create: `driving_algorithm/driving_algorithm/models/__init__.py`
- Create: `driving_algorithm/driving_algorithm/models/cnn_lstm.py`
- Create: `driving_algorithm/tests/test_cnn_lstm.py`

**Interfaces:**
- Produces: `CNNLSTMConfig`
- Produces: `CNNLSTMTrajectoryPredictor.forward(image, state_history, history_mask) -> [B,20,5]`

- [ ] Write tests for output shape, finite values, backward gradients, state-normalisation buffers and rejection of an empty history.
- [ ] Run tests and verify imports fail.
- [ ] Implement ResNet-18 image encoding, per-step state encoding, packed LSTM history, feature fusion and trajectory head.
- [ ] Run tests and commit.

---

### Task 3: Losses and Metrics

**Files:**
- Create: `driving_algorithm/driving_algorithm/training/__init__.py`
- Create: `driving_algorithm/driving_algorithm/training/losses.py`
- Create: `driving_algorithm/driving_algorithm/evaluation/metrics.py`
- Create: `driving_algorithm/driving_algorithm/evaluation/__init__.py`
- Create: `driving_algorithm/tests/test_losses.py`
- Create: `driving_algorithm/tests/test_metrics.py`

**Interfaces:**
- Produces: `trajectory_loss(prediction, target, future_mask, weights) -> dict`
- Produces: `trajectory_metrics(prediction, target, future_mask) -> dict`

- [ ] Write tests for zero perfect-prediction loss, masked corruption, wrapped heading error, ADE and FDE.
- [ ] Run tests and verify imports fail.
- [ ] Implement masked Smooth L1 components and masked trajectory metrics.
- [ ] Run tests and commit.

---

### Task 4: Training, Evaluation and Checkpoint Smoke Test

**Files:**
- Create: `driving_algorithm/driving_algorithm/training/engine.py`
- Create: `driving_algorithm/driving_algorithm/train.py`
- Create: `driving_algorithm/driving_algorithm/evaluate.py`
- Create: `driving_algorithm/tests/test_training_engine.py`
- Modify: `driving_algorithm/README.md`

**Interfaces:**
- Produces: `train_one_epoch(model, loader, optimizer, device, loss_weights) -> dict`
- Produces: `evaluate_model(model, loader, device) -> dict`
- Produces: `save_checkpoint(...)` and `load_checkpoint(...)`

- [ ] Write a tiny synthetic Dataset test that performs a finite optimizer step and checkpoint round trip.
- [ ] Run tests and verify imports fail.
- [ ] Implement training/evaluation loops, atomic checkpoint writing and CLI commands.
- [ ] Run the full test suite.
- [ ] Run one real Waymo batch forward/backward smoke test and document commands.
- [ ] Commit and push the updated PR branch.
