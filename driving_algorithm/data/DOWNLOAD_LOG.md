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
