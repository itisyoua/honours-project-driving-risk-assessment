# Driving Algorithm

This package prepares source-independent driving sequences for the honours
project's CNN-LSTM trajectory predictor. It runs on the Apple Silicon
development Mac without importing or running CARLA.

The common sample timeline is 4 Hz with 16 history steps (4 seconds) and 20
future steps (5 seconds). Raw and converted Waymo data live under `data/` and
are intentionally excluded from Git.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest tests -v
```

Authenticated Waymo download, conversion, validation and preview commands will
be added as their corresponding tools are implemented.
