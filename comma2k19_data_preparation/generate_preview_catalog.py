from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .comma2k19_utils import read_csv_rows
    from .preview_sequence import create_preview
except ImportError:
    from comma2k19_utils import read_csv_rows
    from preview_sequence import create_preview


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
SPEED_BANDS = (("low", 0.1), ("medium", 0.5), ("high", 0.9))


def result_sort_key(path: Path):
    match = re.search(r"chunk_(\d+)_results$", path.name)
    return int(match.group(1)) if match else path.name


def select_representative_rows(rows):
    selected = []
    for split in SPLITS:
        candidates = [
            row
            for row in rows
            if row["split"] == split and int(row["history_start_frame"]) == 0
        ]
        candidates.sort(key=lambda row: float(row["mean_history_speed_mps"]))
        if not candidates:
            raise ValueError(f"No frame-zero candidates for {split}")
        used_sequences = set()
        for band, quantile in SPEED_BANDS:
            index = round((len(candidates) - 1) * quantile)
            offsets = [0]
            for distance in range(1, len(candidates)):
                offsets.extend((-distance, distance))
            for offset in offsets:
                candidate_index = index + offset
                if not 0 <= candidate_index < len(candidates):
                    continue
                row = candidates[candidate_index]
                if row["sequence_id"] not in used_sequences:
                    used_sequences.add(row["sequence_id"])
                    selected.append((split, band, row))
                    break
    return selected


def make_overview(preview_rows, output_path: Path):
    panel_width = 320
    panel_height = 280
    overview = Image.new("RGB", (panel_width * 3, panel_height * 3), "white")
    draw = ImageDraw.Draw(overview)
    for index, item in enumerate(preview_rows):
        image = Image.open(item["preview_png"]).convert("RGB")
        image.thumbnail((panel_width, panel_height - 30))
        x = (index % 3) * panel_width
        y = (index // 3) * panel_height
        overview.paste(image, (x + (panel_width - image.width) // 2, y + 30))
        label = (
            f"{item['split']} / {item['speed_band']} / "
            f"{float(item['mean_history_speed_mps']):.1f} m/s"
        )
        draw.rectangle((x, y, x + panel_width, y + 28), fill=(245, 245, 245))
        draw.text((x + 8, y + 8), label, fill="black")
    overview.save(output_path)


def make_master_overview(result_dirs, output_path: Path):
    panel_width = 480
    panel_height = 440
    columns = 2
    rows = (len(result_dirs) + columns - 1) // columns
    master = Image.new("RGB", (panel_width * columns, panel_height * rows), "white")
    draw = ImageDraw.Draw(master)
    for index, result_dir in enumerate(result_dirs):
        source = result_dir / "previews" / "catalog" / "overview.png"
        image = Image.open(source).convert("RGB")
        image.thumbnail((panel_width, panel_height - 28))
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        master.paste(image, (x + (panel_width - image.width) // 2, y + 28))
        draw.rectangle((x, y, x + panel_width, y + 27), fill=(238, 238, 238))
        draw.text((x + 8, y + 7), result_dir.name.replace("_results", "").upper(), fill="black")
    master.save(output_path)


def write_catalog(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate representative previews for all chunks.")
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    result_dirs = sorted(PREPARATION_DIR.glob("chunk_*_results"), key=result_sort_key)
    data_root = PROJECT_ROOT / "comma2k19"
    total = 0

    if args.overview_only:
        make_master_overview(result_dirs, PREPARATION_DIR / "preview_overview_all_chunks.png")
        print("master_overview:", PREPARATION_DIR / "preview_overview_all_chunks.png")
        return

    for result_dir in result_dirs:
        chunk_number = int(re.search(r"chunk_(\d+)_results$", result_dir.name).group(1))
        index_path = result_dir / f"comma2k19_chunk_{chunk_number}_sequence_index.csv"
        selected = select_representative_rows(read_csv_rows(index_path))
        catalog_root = result_dir / "previews" / "catalog"
        catalog_root.mkdir(parents=True, exist_ok=True)
        catalog_rows = []

        for split, band, row in selected:
            output_dir = catalog_root / split / band
            png_path, gif_path, metadata = create_preview(row, data_root, output_dir)
            catalog_rows.append(
                {
                    "chunk": row["chunk"],
                    "split": split,
                    "speed_band": band,
                    "mean_history_speed_mps": row["mean_history_speed_mps"],
                    "route_id": row["route_id"],
                    "segment_id": row["segment_id"],
                    "sequence_id": row["sequence_id"],
                    "target_end_longitudinal_m": metadata["target_end_longitudinal_m"],
                    "target_end_lateral_m": metadata["target_end_lateral_m"],
                    "preview_png": str(png_path),
                    "preview_gif": str(gif_path),
                }
            )

        write_catalog(catalog_root / "preview_catalog.csv", catalog_rows)
        make_overview(catalog_rows, catalog_root / "overview.png")
        total += len(catalog_rows)
        print(f"{result_dir.name}: {len(catalog_rows)} previews")

    make_master_overview(result_dirs, PREPARATION_DIR / "preview_overview_all_chunks.png")
    print(f"total_previews: {total}")


if __name__ == "__main__":
    main()
