# Waymo E2E Download Log

## 2026-08-21 Validation Shard

- Dataset: Waymo Open Dataset, End-to-End Driving Dataset V1.0.0
- Split: validation
- File: `val_202504211843.tfrecord-00000-of-00093`
- Size: 2,615,071,793 bytes
- SHA-256: `f54c97cbc9fdfaed2b731939856e7af182c10c161619054b5030bf16d15231cb`
- Source: authenticated official Google Cloud Storage download reached from <https://waymo.com/open/download/>
- Dataset terms: <https://waymo.com/open/terms/>
- Local raw path: `data/raw/waymo_e2e/val_202504211843.tfrecord-00000-of-00093`
- Inspection: 1,150 records, 425 route prefixes, zero malformed records, zero duplicate frame IDs
- Compatibility: passed; all 1,150 records contain a current `FRONT` image, complete 16-step past ego state and complete 20-step future x/y target
- Note: this E2E release stores one current camera image per native sample. Its 16 temporal inputs are the provided 4 Hz ego-state history, not a 16-image sequence.
- Conversion: 1,150 model samples, zero rejected records, 425 route prefixes
- Converted size: 373,234,894 bytes referenced by the validation manifest
- Validation: passed with zero duplicate IDs, route leakage, missing files, decode errors, shape errors or non-finite values

## 2026-08-21 Training Subset

- Dataset: Waymo Open Dataset, End-to-End Driving Dataset V1.0.0
- Split: training
- Source: authenticated official Google Cloud Storage download reached from <https://waymo.com/open/download/>
- Selected shards: `00000` through `00006` of 263

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `training_202504031202_202504151040.tfrecord-00000-of-00263` | 3,604,761,105 | `c67f8b458dca5d0be9b21e9b9074679f7a0bf8a8fd9ad33b1e3fee7d1f1b43ae` |
| `training_202504031202_202504151040.tfrecord-00001-of-00263` | 3,416,257,215 | `633d0b497fb045a5c1f3c486320574e5bb5897244a3400f5e7e1066d0363efbd` |
| `training_202504031202_202504151040.tfrecord-00002-of-00263` | 3,536,679,290 | `5a64691884091911e1010cc16a97d695317d85373c189330e94922d9b7e221b7` |
| `training_202504031202_202504151040.tfrecord-00003-of-00263` | 3,596,195,561 | `5eaa275f1cbc507e02602feeedc50fc88d49a73d37c648eb50f74675729511c2` |
| `training_202504031202_202504151040.tfrecord-00004-of-00263` | 3,688,407,900 | `e70acbd647a5e44e3bba4b59ce38ca144f16c34a7a35fcb6ed2d23aab359de95` |
| `training_202504031202_202504151040.tfrecord-00005-of-00263` | 3,454,925,906 | `9ca9bca4e0756524afc2c76b9bec539f1afd3a97b29c36849676db8c4ead1dae` |
| `training_202504031202_202504151040.tfrecord-00006-of-00263` | 3,569,787,442 | `55d44ab197a349b644de93a2ddd1273fd41d2c8d847c1c60e25238db2a43cc40` |

- Inspection: 10,979 records, 2,022 route prefixes, zero malformed records and zero duplicate frame IDs
- Compatibility: passed for all 10,979 records
- Conversion: 10,979 model samples, zero rejected records
- Converted size: 3,520,549,816 bytes referenced by the training manifest
- Validation: passed with zero errors
- Intent counts: 914 `go_left`, 814 `go_right`, 9,251 `go_straight`

## Prepared Subset Summary

- Official raw shards: 8 total, 27,482,086,212 bytes
- Converted images and motion arrays: 3,893,784,710 bytes
- Raw plus converted total: 31,375,870,922 bytes (about 29.2 GiB)
- Model samples: 12,129 total (10,979 train and 1,150 validation)
- Route prefixes: 2,447 total (2,022 train and 425 validation)
- Cross-split route overlap: zero
- Cross-split sample-ID overlap: zero
- Download stopped here to remain within the approved 20-40 GB storage budget.
