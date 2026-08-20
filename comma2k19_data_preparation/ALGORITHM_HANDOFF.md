# Algorithm Handoff Contract

## Model input

Each item returned by `Comma2k19Dataset` contains:

| Key | Shape | Description |
| --- | --- | --- |
| `frames` | `[30, 3, 224, 224]` | ImageNet-normalised RGB history frames |
| `state_history` | `[30, 8]` | Normalised ego-vehicle motion history |
| `future_target` | `[20, 5]` | Raw future motion target in local coordinates |
| `sequence_id` | string | Stable sample identifier |
| `route_id` | string | Source route identifier |
| `split` | string | `train`, `validation` or `test` |

CNN features should be extracted for each of the 30 frames. Concatenate those
features with the corresponding eight state features before passing the
sequence to the LSTM. The prediction head should output 20 time steps with five
values per time step.

The default constructor uses the combined Chunk_1 through Chunk_10 index:
234,650 sequences across 178 independent routes.

## Split policy

Use the provided split column or the files under `chunk_1_results/splits`.
Never randomly split individual rows because adjacent sliding windows overlap.
All samples from one `route_id` stay in exactly one split.

## Expected evaluation

- Position: Average Displacement Error and Final Displacement Error
- Speed: MAE or RMSE
- Heading: angular MAE
- Risk stage in CARLA: classification precision, recall, F1 and warning lead time

## Boundary with CARLA

This dataset trains the visual and ego-motion prediction components. CARLA must
provide surrounding-actor position, velocity, dimensions, mass, collision
events and scenario metadata for MTTC, collision probability and severity.
