"""
Trajectory Dataset Setup Pipeline
For autonomous driving risk assessment and CARLA simulation preparation.

This version only loads the two usable trajectory datasets:
1. Next Generation Simulation (NGSIM)
2. WholeVdata2

Output:
processed_data/usable_carla_ready_trajectories.csv
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# BASIC SETTINGS
# ============================================================

RAW_DATA_PATH = "./raw_data"
OUTPUT_PATH = "./processed_data"

TARGET_FILES = [
    "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv",
    "WholeVdata2.csv"
]

MIN_TRAJECTORY_LENGTH = 10
MAX_REASONABLE_SPEED = 70.0
MAX_REASONABLE_ACCEL = 15.0
MAX_TIME_GAP = 2.0
DEFAULT_FRAME_INTERVAL = 0.1


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_MAPPING = {
    "vehicle_id": "vehicle_id",
    "frame_id": "frame_id",
    "global_time": "time",

    "local_x": "x",
    "local_y": "y",

    "v_vel": "speed",
    "v_acc": "acceleration",

    "lane_id": "lane_id",
    "v_class": "vehicle_type"
}


def normalise_column_name(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def standardise_columns(df):
    df = df.copy()
    df.columns = [normalise_column_name(c) for c in df.columns]
    df = df.rename(columns=COLUMN_MAPPING)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_single_file(file_path):
    try:
        df = pd.read_csv(
            file_path,
            on_bad_lines="skip",
            engine="c",
            low_memory=False
        )

        df["source_file"] = file_path.name
        return df

    except Exception as e:
        print(f"Failed to load {file_path.name}: {e}")
        return None


def load_all_files(raw_data_path):
    raw_path = Path(raw_data_path)

    if not raw_path.exists():
        raise FileNotFoundError("raw_data folder not found.")

    files = [
        p for p in raw_path.rglob("*.csv")
        if p.name in TARGET_FILES
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            "No target trajectory files found. Please check raw_data folder."
        )

    all_dfs = []

    for file in files:
        print(f"Loading file: {file.name}")
        temp_df = load_single_file(file)

        if temp_df is not None:
            all_dfs.append(temp_df)

    if len(all_dfs) == 0:
        raise RuntimeError("No files were loaded successfully.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nLoaded rows: {len(combined_df)}")
    print(f"Loaded files: {len(all_dfs)}")

    return combined_df


# ============================================================
# VALIDATION AND CLEANING
# ============================================================

def check_required_columns(df):
    required = ["vehicle_id", "frame_id", "time", "x", "y"]

    missing = [c for c in required if c not in df.columns]

    if len(missing) > 0:
        print("\nAvailable columns:")
        for col in df.columns:
            print(f" - {col}")

        raise ValueError(f"Missing required columns: {missing}")


def convert_numeric_columns(df):
    df = df.copy()

    numeric_columns = [
        "vehicle_id", "frame_id", "time", "x", "y",
        "speed", "acceleration", "lane_id", "vehicle_type"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_invalid_rows(df):
    df = df.copy()

    df = df.dropna(subset=["vehicle_id", "frame_id", "time", "x", "y"])
    df = df.drop_duplicates()

    return df


def sort_data(df):
    df = df.copy()
    df = df.sort_values(["vehicle_id", "time"])
    df = df.reset_index(drop=True)
    return df


# ============================================================
# FEATURE GENERATION
# ============================================================

def generate_motion_features(df):
    df = df.copy()

    df["dt"] = df.groupby("vehicle_id")["time"].diff()

    # NGSIM global_time is usually in milliseconds
    if df["dt"].median() > 10:
        df["time"] = df["time"] / 1000.0
        df["dt"] = df.groupby("vehicle_id")["time"].diff()

    df.loc[df["dt"] <= 0, "dt"] = np.nan

    df["dx"] = df.groupby("vehicle_id")["x"].diff()
    df["dy"] = df.groupby("vehicle_id")["y"].diff()

    df["vx"] = df["dx"] / df["dt"]
    df["vy"] = df["dy"] / df["dt"]

    if "speed" not in df.columns:
        df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)

    if "acceleration" not in df.columns:
        df["acceleration"] = df.groupby("vehicle_id")["speed"].diff() / df["dt"]

    df["heading_rad"] = np.arctan2(df["vy"], df["vx"])
    df["heading_deg"] = np.degrees(df["heading_rad"])

    return df


# ============================================================
# QUALITY CHECK
# ============================================================

def dataset_summary(df):
    records = []

    for source_file, group in df.groupby("source_file"):
        records.append({
            "source_file": source_file,
            "rows": len(group),
            "vehicles": group["vehicle_id"].nunique(),
            "time_min": group["time"].min(),
            "time_max": group["time"].max(),
            "duration": group["time"].max() - group["time"].min(),
            "missing_rate": group.isna().mean().mean()
        })

    return pd.DataFrame(records)


def trajectory_quality_report(df):
    records = []

    for vehicle_id, traj in df.groupby("vehicle_id"):
        traj = traj.sort_values("time")

        n_points = len(traj)
        duration = traj["time"].max() - traj["time"].min()
        max_time_gap = traj["dt"].max()
        max_speed = traj["speed"].max()
        mean_speed = traj["speed"].mean()
        max_accel = traj["acceleration"].abs().max()
        missing_rate = traj.isna().mean().mean()

        usable = True
        reasons = []

        if n_points < MIN_TRAJECTORY_LENGTH:
            usable = False
            reasons.append("trajectory_too_short")

        if pd.notna(max_time_gap) and max_time_gap > MAX_TIME_GAP:
            usable = False
            reasons.append("large_time_gap")

        if pd.notna(max_speed) and max_speed > MAX_REASONABLE_SPEED:
            usable = False
            reasons.append("unreasonable_speed")

        if pd.notna(max_accel) and max_accel > MAX_REASONABLE_ACCEL:
            usable = False
            reasons.append("unreasonable_acceleration")

        if missing_rate > 0.3:
            usable = False
            reasons.append("too_many_missing_values")

        records.append({
            "vehicle_id": vehicle_id,
            "source_file": traj["source_file"].iloc[0],
            "n_points": n_points,
            "duration": duration,
            "max_time_gap": max_time_gap,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "max_abs_acceleration": max_accel,
            "missing_rate": missing_rate,
            "usable": usable,
            "reason": "; ".join(reasons) if reasons else "valid"
        })

    return pd.DataFrame(records)


# ============================================================
# FILTER AND EXPORT
# ============================================================

def filter_usable_data(df, quality_df):
    usable_ids = quality_df.loc[quality_df["usable"] == True, "vehicle_id"]
    usable_df = df[df["vehicle_id"].isin(usable_ids)].copy()
    return usable_df.reset_index(drop=True)


def select_output_columns(df):
    output_columns = [
        "source_file",
        "vehicle_id",
        "frame_id",
        "time",
        "x",
        "y",
        "vx",
        "vy",
        "speed",
        "acceleration",
        "heading_rad",
        "heading_deg",
        "lane_id",
        "vehicle_type"
    ]

    existing_columns = [c for c in output_columns if c in df.columns]
    return df[existing_columns].copy()


def add_nearest_vehicle_features(df):
    df = df.copy()

    df["nearest_vehicle_id"] = np.nan
    df["nearest_distance"] = np.nan

    results = []

    for time_value, frame_data in df.groupby("time"):
        frame_data = frame_data.copy()

        if len(frame_data) <= 1:
            results.append(frame_data)
            continue

        positions = frame_data[["x", "y"]].to_numpy()
        vehicle_ids = frame_data["vehicle_id"].to_numpy()

        nearest_ids = []
        nearest_distances = []

        for i in range(len(frame_data)):
            diff = positions - positions[i]
            distance = np.sqrt(np.sum(diff ** 2, axis=1))
            distance[i] = np.inf

            nearest_index = np.argmin(distance)
            nearest_ids.append(vehicle_ids[nearest_index])
            nearest_distances.append(distance[nearest_index])

        frame_data["nearest_vehicle_id"] = nearest_ids
        frame_data["nearest_distance"] = nearest_distances

        results.append(frame_data)

    return pd.concat(results, ignore_index=True)


def export_files(raw_df, processed_df, usable_df, summary_df, quality_df):
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(output_path / "all_processed_trajectories.csv", index=False)
    usable_df.to_csv(output_path / "usable_carla_ready_trajectories.csv", index=False)
    summary_df.to_csv(output_path / "dataset_summary.csv", index=False)
    quality_df.to_csv(output_path / "trajectory_quality_report.csv", index=False)

    report = {
        "raw_rows": int(len(raw_df)),
        "processed_rows": int(len(processed_df)),
        "usable_rows": int(len(usable_df)),
        "raw_vehicle_count": int(processed_df["vehicle_id"].nunique()),
        "usable_vehicle_count": int(usable_df["vehicle_id"].nunique()),
        "main_output": "usable_carla_ready_trajectories.csv"
    }

    with open(output_path / "processing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nExport completed.")
    print(f"Output folder: {output_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("==============================================")
    print("Trajectory Dataset Setup Pipeline")
    print("==============================================")

    raw_df = load_all_files(RAW_DATA_PATH)

    df = standardise_columns(raw_df)
    check_required_columns(df)

    df = convert_numeric_columns(df)
    df = remove_invalid_rows(df)
    df = sort_data(df)

    df = generate_motion_features(df)

    df = df[df["speed"].abs() <= MAX_REASONABLE_SPEED]
    df = df[df["acceleration"].abs() <= MAX_REASONABLE_ACCEL]
    df = df.reset_index(drop=True)

    summary_df = dataset_summary(df)
    quality_df = trajectory_quality_report(df)

    usable_df = filter_usable_data(df, quality_df)
    usable_df = select_output_columns(usable_df)
    usable_df = add_nearest_vehicle_features(usable_df)

    export_files(raw_df, df, usable_df, summary_df, quality_df)

    print("\n==============================================")
    print("Processing Summary")
    print("==============================================")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Processed rows: {len(df)}")
    print(f"Usable rows: {len(usable_df)}")
    print(f"Usable vehicles: {usable_df['vehicle_id'].nunique()}")
    print("\nMain file for next step:")
    print("processed_data/usable_carla_ready_trajectories.csv")
    print("==============================================")


if __name__ == "__main__":
    main()