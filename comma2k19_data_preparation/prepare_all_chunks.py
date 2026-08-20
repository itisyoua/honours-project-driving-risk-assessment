from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_DIR = Path(__file__).resolve().parent


def chunk_sort_key(path: Path):
    match = re.search(r"Chunk_(\d+)$", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def run(command):
    print("Running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare every extracted comma2k19 chunk.")
    parser.add_argument("--chunks-root", default=str(PROJECT_ROOT / "comma2k19"))
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--prediction-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    chunk_paths = sorted(
        (path for path in Path(args.chunks_root).glob("Chunk_*") if path.is_dir()),
        key=chunk_sort_key,
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No Chunk_* directories found in {args.chunks_root}")

    for chunk_path in chunk_paths:
        result_dir = PREPARATION_DIR / f"{chunk_path.name.lower()}_results"
        summary_path = result_dir / f"comma2k19_{chunk_path.name.lower()}_summary.json"
        if args.skip_existing and summary_path.exists():
            print("Skipping existing:", chunk_path.name, flush=True)
            continue
        run(
            [
                sys.executable,
                PREPARATION_DIR / "comma2k19_manifest.py",
                "--chunk",
                chunk_path,
                "--history-len",
                args.history_len,
                "--prediction-len",
                args.prediction_len,
                "--stride",
                args.stride,
            ]
        )

    run([sys.executable, PREPARATION_DIR / "combine_chunk_results.py"])
    run([sys.executable, PREPARATION_DIR / "validate_preparation.py"])


if __name__ == "__main__":
    main()
