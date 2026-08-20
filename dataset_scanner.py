from pathlib import Path
import pandas as pd

RAW_DATA_PATH = "./raw_data"
OUTPUT_PATH = "./processed_data"

REQUIRED_GROUPS = {
    "vehicle_id": ["vehicle_id", "veh_id", "track_id", "id", "object_id", "agent_id"],
    "x": ["x", "pos_x", "position_x", "local_x", "global_x", "world_x"],
    "y": ["y", "pos_y", "position_y", "local_y", "global_y", "world_y"],
    "time_or_frame": ["time", "timestamp", "t", "frame", "frame_id", "frame_number"]
}

EXCLUDE_KEYWORDS = [
    "collision",
    "crash",
    "specification",
    "specifications",
    "veh_spec",
    "vehicle_spec",
    "adas"
]


def normalise(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def detect_columns(columns):
    columns = [normalise(c) for c in columns]

    detected = {}

    for key, candidates in REQUIRED_GROUPS.items():
        detected[key] = any(c in columns for c in candidates)

    return detected


def scan_file(file_path):
    result = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "usable": False,
        "reason": "",
        "rows_checked": 0,
        "columns": ""
    }

    lower_name = file_path.name.lower()

    if any(keyword in lower_name for keyword in EXCLUDE_KEYWORDS):
        result["reason"] = "excluded_by_file_name"
        return result

    try:
        df = pd.read_csv(
            file_path,
            nrows=50,
            on_bad_lines="skip",
            engine="c",
            low_memory=True
        )

        result["rows_checked"] = len(df)
        result["columns"] = ", ".join([normalise(c) for c in df.columns])

        if df.shape[1] < 4:
            result["reason"] = "too_few_columns"
            return result

        detected = detect_columns(df.columns)

        if detected["vehicle_id"] and detected["x"] and detected["y"] and detected["time_or_frame"]:
            result["usable"] = True
            result["reason"] = "usable_trajectory_file"
        else:
            missing = [k for k, v in detected.items() if not v]
            result["reason"] = "missing_" + "_".join(missing)

        return result

    except Exception as e:
        result["reason"] = "read_failed"
        return result


def main():
    raw_path = Path(RAW_DATA_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in raw_path.rglob("*.csv")]

    results = []

    print("Scanning CSV files...")

    for file in files:
        print(f"Checking: {file.name}")
        results.append(scan_file(file))

    report_df = pd.DataFrame(results)

    report_df.to_csv(output_path / "dataset_scan_report.csv", index=False)

    usable_df = report_df[report_df["usable"] == True]
    unusable_df = report_df[report_df["usable"] == False]

    usable_df.to_csv(output_path / "usable_dataset_files.csv", index=False)
    unusable_df.to_csv(output_path / "excluded_dataset_files.csv", index=False)

    print("\nScan completed.")
    print(f"Total CSV files: {len(report_df)}")
    print(f"Usable trajectory files: {len(usable_df)}")
    print(f"Excluded files: {len(unusable_df)}")
    print("\nOutput files:")
    print("processed_data/dataset_scan_report.csv")
    print("processed_data/usable_dataset_files.csv")
    print("processed_data/excluded_dataset_files.csv")


if __name__ == "__main__":
    main()