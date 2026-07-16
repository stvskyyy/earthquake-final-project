from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "model_dataset_spatial.csv"
OUTPUT_PATH = PROCESSED_DIR / "model_dataset_spatial_clean.csv"
REPORT_PATH = OUTPUT_TABLES_DIR / "phase3_final_model_dataset_report.txt"


def main():
    print("Starting Phase 3B: Final Spatial Model Dataset Cleanup")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    original_shape = df.shape
    original_rows = len(df)

    report_lines = []
    report_lines.append("PHASE 3B FINAL SPATIAL MODEL DATASET CLEANUP REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Original shape: {original_shape}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 1. Remove no-data / non-soil rows
    # ------------------------------------------------------------
    no_data_soil_values = [
        "Water body (includes no data)",
        "water body (includes no data)",
        "WATER BODY (INCLUDES NO DATA)",
    ]

    before_remove_no_data = len(df)

    if "soil_desc" in df.columns:
        df = df[~df["soil_desc"].isin(no_data_soil_values)].copy()

    after_remove_no_data = len(df)
    removed_no_data = before_remove_no_data - after_remove_no_data

    report_lines.append("1. REMOVED NON-SOIL / NO-DATA RECORDS")
    report_lines.append("-" * 80)
    report_lines.append(f"Rows before removal: {before_remove_no_data}")
    report_lines.append(f"Rows after removal: {after_remove_no_data}")
    report_lines.append(f"Rows removed: {removed_no_data}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 2. Drop columns not recommended for baseline modeling
    # ------------------------------------------------------------
    columns_to_drop = [
        # High missingness seismic quality columns
        "nst",
        "dmin",
        "gap",

        # Optional: rms has some missing values. We keep it out for baseline simplicity.
        "rms",

        # Non-feature / identifier columns
        "id",
        "updated",

        # Place is textual and messy; useful for reporting but not baseline model.
        "place",

        # type_property is likely constant or not useful if all are earthquake.
        "type_property",
    ]

    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns_to_drop, errors="ignore")

    report_lines.append("2. DROPPED COLUMNS")
    report_lines.append("-" * 80)
    if existing_columns_to_drop:
        for col in existing_columns_to_drop:
            report_lines.append(f"- {col}")
    else:
        report_lines.append("No columns dropped.")
    report_lines.append("")

    # ------------------------------------------------------------
    # 3. Clean text columns
    # ------------------------------------------------------------
    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # ------------------------------------------------------------
    # 4. Convert time column and create clean temporal fields
    # ------------------------------------------------------------
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    before_time_drop = len(df)

    if "time" in df.columns:
        df = df.dropna(subset=["time"]).copy()

    after_time_drop = len(df)
    removed_invalid_time = before_time_drop - after_time_drop

    # Recreate time features just to make sure they are consistent.
    if "time" in df.columns:
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek

    report_lines.append("3. TIME CLEANING")
    report_lines.append("-" * 80)
    report_lines.append(f"Rows removed due to invalid time: {removed_invalid_time}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 5. Remove rows with missing required modeling fields
    # ------------------------------------------------------------
    required_model_cols = [
        "mag",
        "longitude",
        "latitude",
        "depth",
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "magtype",
        "tsunami",
        "soil_desc",
        "soil_series",
        "aez",
        "soil_polygon_hectares",
    ]

    existing_required_cols = [col for col in required_model_cols if col in df.columns]
    missing_required_cols = [col for col in required_model_cols if col not in df.columns]

    before_required_drop = len(df)
    df = df.dropna(subset=existing_required_cols).copy()
    after_required_drop = len(df)

    removed_missing_required = before_required_drop - after_required_drop

    report_lines.append("4. REQUIRED MODELING FIELDS")
    report_lines.append("-" * 80)
    report_lines.append("Required columns checked:")
    for col in existing_required_cols:
        report_lines.append(f"- {col}")

    if missing_required_cols:
        report_lines.append("")
        report_lines.append("WARNING: These expected columns were not found:")
        for col in missing_required_cols:
            report_lines.append(f"- {col}")

    report_lines.append("")
    report_lines.append(f"Rows before dropping missing required fields: {before_required_drop}")
    report_lines.append(f"Rows after dropping missing required fields: {after_required_drop}")
    report_lines.append(f"Rows removed: {removed_missing_required}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 6. Remove exact duplicates
    # ------------------------------------------------------------
    before_duplicates = len(df)
    df = df.drop_duplicates()
    after_duplicates = len(df)
    removed_duplicates = before_duplicates - after_duplicates

    report_lines.append("5. DUPLICATE CLEANING")
    report_lines.append("-" * 80)
    report_lines.append(f"Rows before duplicate removal: {before_duplicates}")
    report_lines.append(f"Rows after duplicate removal: {after_duplicates}")
    report_lines.append(f"Rows removed: {removed_duplicates}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 7. Save clean modeling dataset
    # ------------------------------------------------------------
    final_shape = df.shape
    final_rows = len(df)

    df.to_csv(OUTPUT_PATH, index=False)

    report_lines.append("6. FINAL DATASET SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Original rows: {original_rows}")
    report_lines.append(f"Final rows: {final_rows}")
    report_lines.append(f"Final shape: {final_shape}")
    report_lines.append(f"Saved clean model dataset to: {OUTPUT_PATH}")
    report_lines.append("")

    report_lines.append("Final columns:")
    for col in df.columns:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Missing values in final dataset:")
    report_lines.append(df.isna().sum().to_string())
    report_lines.append("")

    report_lines.append("Magnitude summary:")
    if "mag" in df.columns:
        report_lines.append(df["mag"].describe().to_string())
    else:
        report_lines.append("mag column not found.")
    report_lines.append("")

    report_lines.append("Depth summary:")
    if "depth" in df.columns:
        report_lines.append(df["depth"].describe().to_string())
    else:
        report_lines.append("depth column not found.")
    report_lines.append("")

    report_lines.append("Top soil descriptions:")
    if "soil_desc" in df.columns:
        report_lines.append(df["soil_desc"].value_counts().head(20).to_string())
    else:
        report_lines.append("soil_desc column not found.")
    report_lines.append("")

    report_lines.append("Top soil series:")
    if "soil_series" in df.columns:
        report_lines.append(df["soil_series"].value_counts().head(20).to_string())
    else:
        report_lines.append("soil_series column not found.")
    report_lines.append("")

    report_lines.append("Top AEZ values:")
    if "aez" in df.columns:
        report_lines.append(df["aez"].value_counts().head(20).to_string())
    else:
        report_lines.append("aez column not found.")
    report_lines.append("")

    report_lines.append("7. MODELING NOTE")
    report_lines.append("-" * 80)
    report_lines.append(
        "This dataset is intended for the baseline machine learning model. "
        "Columns with severe missingness, non-soil/no-data polygons, and likely non-useful identifier fields were removed. "
        "The target variable is mag. Soil attributes are treated as contextual spatial predictors, not direct causal factors of earthquake magnitude."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Phase 3B complete.")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Saved clean model dataset to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()