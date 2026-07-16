from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


EARTHQUAKE_FILE = "KODA_Datasets - Philippine_Earthquakes_USGS.csv"

SOIL_FILES = [
    "KODA_Datasets - Agusan Soil Data.csv",
    "KODA_Datasets - Bukidnon Soil Data.csv",
    "KODA_Datasets - Cotabato Soil Data.csv",
    "KODA_Datasets - Davao Soil Data.csv",
    "KODA_Datasets - Misamis Occidental Soil Data.csv",
    "KODA_Datasets - Misamis Oriental Soil Data.csv",
    "KODA_Datasets - Surigao Del Norte Soil Data.csv",
    "KODA_Datasets - Surigao Del Sur Soil Data.csv",
    "KODA_Datasets - Zamboanga Del Norte Soil Data.csv",
]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    missing_like = [
        "",
        "nan",
        "none",
        "null",
        "unknown",
        "UNKNOWN",
        "not avail.",
        "Not Avail.",
        "NOT AVAIL.",
        "not avail",
        "Not Avail",
        "NOT AVAIL",
        "n/a",
        "N/A",
        "...",
    ]

    for col in text_cols:
        df[col] = df[col].replace(missing_like, np.nan)

    return df


def drop_empty_and_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns_to_drop = []

    for col in df.columns:
        lower_col = col.lower()

        if lower_col.startswith("unnamed"):
            columns_to_drop.append(col)

        if lower_col.startswith("source:"):
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def clean_earthquake_data() -> tuple[pd.DataFrame, dict]:
    path = RAW_DIR / EARTHQUAKE_FILE

    if not path.exists():
        raise FileNotFoundError(f"Earthquake file not found: {path}")

    raw_eq = pd.read_csv(path)
    original_shape = raw_eq.shape

    eq = standardize_column_names(raw_eq)

    # After standardization, magType becomes magtype.
    useful_cols = [
        "id",
        "mag",
        "place",
        "time",
        "updated",
        "tsunami",
        "sig",
        "nst",
        "dmin",
        "rms",
        "gap",
        "magtype",
        "type_property",
        "title",
        "longitude",
        "latitude",
        "depth",
    ]

    available_cols = [col for col in useful_cols if col in eq.columns]
    eq = eq[available_cols].copy()

    # Convert datetime fields.
    if "time" in eq.columns:
        eq["time"] = pd.to_datetime(eq["time"], errors="coerce")

    if "updated" in eq.columns:
        eq["updated"] = pd.to_datetime(eq["updated"], errors="coerce")

    before_drop_essential = len(eq)

    essential_cols = ["mag", "latitude", "longitude", "depth", "time"]
    existing_essential_cols = [col for col in essential_cols if col in eq.columns]

    eq = eq.dropna(subset=existing_essential_cols)

    after_drop_essential = len(eq)

    # Filter suspicious/impossible values.
    before_filter = len(eq)

    if "mag" in eq.columns:
        eq = eq[eq["mag"] > 0]

    if "latitude" in eq.columns:
        eq = eq[eq["latitude"].between(4, 22)]

    if "longitude" in eq.columns:
        eq = eq[eq["longitude"].between(116, 128)]

    if "depth" in eq.columns:
        eq = eq[eq["depth"] >= 0]

    after_filter = len(eq)

    # Create time features.
    eq["year"] = eq["time"].dt.year
    eq["month"] = eq["time"].dt.month
    eq["day"] = eq["time"].dt.day
    eq["hour"] = eq["time"].dt.hour
    eq["day_of_week"] = eq["time"].dt.dayofweek

    # Create simple magnitude category.
    eq["magnitude_category"] = pd.cut(
        eq["mag"],
        bins=[0, 3.9, 4.9, 5.9, 6.9, 10],
        labels=["minor", "light", "moderate", "strong", "major"],
        include_lowest=True,
    )

    eq = eq.sort_values("time").reset_index(drop=True)

    output_path = PROCESSED_DIR / "clean_earthquakes.csv"
    eq.to_csv(output_path, index=False)

    report = {
        "raw_shape": original_shape,
        "clean_shape": eq.shape,
        "rows_before_drop_essential": before_drop_essential,
        "rows_after_drop_essential": after_drop_essential,
        "rows_removed_missing_essential": before_drop_essential - after_drop_essential,
        "rows_before_filter": before_filter,
        "rows_after_filter": after_filter,
        "rows_removed_suspicious_values": before_filter - after_filter,
        "output_path": output_path,
    }

    return eq, report

def safe_mode(series):
    valid = series.dropna()
    return valid.mode().iloc[0] if not valid.mode().empty else np.nan


def clean_and_combine_soil_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    soil_frames = []
    file_reports = []

    standard_soil_cols = [
        "descrip",
        "mapdate",
        "mapscale",
        "province",
        "category",
        "source",
        "maptype",
        "class",
        "shape_leng",
        "shape_area",
    ]

    for file_name in SOIL_FILES:
        path = RAW_DIR / file_name

        if not path.exists():
            print(f"WARNING: Soil file not found: {path}")
            continue

        raw_df = pd.read_csv(path)
        original_shape = raw_df.shape

        df = standardize_column_names(raw_df)
        df = drop_empty_and_source_columns(df)

        # Drop Davao-specific duplicate/reference columns.
        df = df.drop(
            columns=["oid_", "class_1", "category_1", "maptype_1"],
            errors="ignore",
        )

        # Keep only columns available from the standard set.
        available_cols = [col for col in standard_soil_cols if col in df.columns]
        df = df[available_cols].copy()

        df = clean_text_columns(df)

        # Standardize selected text columns.
        for col in ["descrip", "province", "category", "source", "maptype"]:
            if col in df.columns:
                df[col] = df[col].str.strip()

        if "maptype" in df.columns:
            df["maptype"] = df["maptype"].str.upper()

        if "province" in df.columns:
            df["province"] = df["province"].str.title()

        if "category" in df.columns:
            df["category"] = df["category"].str.upper()

        if "descrip" in df.columns:
            df["descrip"] = df["descrip"].str.upper()

        # Make class numeric where possible.
        if "class" in df.columns:
            df["class"] = pd.to_numeric(df["class"], errors="coerce")

        # Keep traceability.
        df["source_file"] = file_name

        before_duplicates = len(df)
        df = df.drop_duplicates()
        after_duplicates = len(df)

        soil_frames.append(df)

        file_reports.append({
            "file_name": file_name,
            "raw_shape": original_shape,
            "clean_shape": df.shape,
            "duplicates_removed": before_duplicates - after_duplicates,
        })

    if not soil_frames:
        raise ValueError("No soil files were loaded. Check SOIL_FILES and data/raw folder.")

    soil = pd.concat(soil_frames, ignore_index=True)

    before_global_duplicates = len(soil)
    soil = soil.drop_duplicates()
    after_global_duplicates = len(soil)

    output_path = PROCESSED_DIR / "combined_soil_data.csv"
    soil.to_csv(output_path, index=False)

    soil_summary = (
        soil
        .groupby("province", dropna=False)
        .agg(
            dominant_soil_category=("category", safe_mode),
            dominant_soil_description=("descrip", safe_mode),
            soil_class_mode=("class", safe_mode),
            soil_type_count=("category", "nunique"),
            record_count=("category", "count"),
        )
        .reset_index()
    )

    summary_output_path = PROCESSED_DIR / "soil_summary_by_province.csv"
    soil_summary.to_csv(summary_output_path, index=False)

    report = {
        "file_reports": file_reports,
        "combined_soil_shape": soil.shape,
        "global_duplicates_removed": before_global_duplicates - after_global_duplicates,
        "combined_soil_output_path": output_path,
        "soil_summary_shape": soil_summary.shape,
        "soil_summary_output_path": summary_output_path,
    }

    return soil, soil_summary, report


def write_cleaning_report(
    earthquake_report: dict,
    soil_report: dict,
    clean_earthquakes: pd.DataFrame,
    combined_soil: pd.DataFrame,
    soil_summary: pd.DataFrame,
) -> None:
    report_path = OUTPUT_TABLES_DIR / "phase2_cleaning_report.txt"

    lines = []

    lines.append("PHASE 2 DATA CLEANING REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. EARTHQUAKE DATA CLEANING")
    lines.append("-" * 80)
    lines.append(f"Raw earthquake shape: {earthquake_report['raw_shape']}")
    lines.append(f"Clean earthquake shape: {earthquake_report['clean_shape']}")
    lines.append(
        "Rows removed due to missing essential values "
        f"(mag, latitude, longitude, depth, time): "
        f"{earthquake_report['rows_removed_missing_essential']}"
    )
    lines.append(
        "Rows removed due to suspicious/impossible values: "
        f"{earthquake_report['rows_removed_suspicious_values']}"
    )
    lines.append(f"Saved to: {earthquake_report['output_path']}")
    lines.append("")

    lines.append("Clean earthquake columns:")
    for col in clean_earthquakes.columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("Clean earthquake missing values:")
    lines.append(clean_earthquakes.isna().sum().to_string())
    lines.append("")

    lines.append("2. SOIL DATA CLEANING")
    lines.append("-" * 80)

    for file_report in soil_report["file_reports"]:
        lines.append(f"File: {file_report['file_name']}")
        lines.append(f"Raw shape: {file_report['raw_shape']}")
        lines.append(f"Clean shape: {file_report['clean_shape']}")
        lines.append(f"Duplicates removed: {file_report['duplicates_removed']}")
        lines.append("")

    lines.append(f"Combined soil shape: {soil_report['combined_soil_shape']}")
    lines.append(f"Global duplicates removed after concatenation: {soil_report['global_duplicates_removed']}")
    lines.append(f"Saved combined soil data to: {soil_report['combined_soil_output_path']}")
    lines.append("")

    lines.append("Combined soil columns:")
    for col in combined_soil.columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("Combined soil missing values:")
    lines.append(combined_soil.isna().sum().to_string())
    lines.append("")

    lines.append("3. SOIL SUMMARY BY PROVINCE")
    lines.append("-" * 80)
    lines.append(f"Soil summary shape: {soil_report['soil_summary_shape']}")
    lines.append(f"Saved soil summary to: {soil_report['soil_summary_output_path']}")
    lines.append("")

    lines.append("Soil summary preview:")
    lines.append(soil_summary.to_string(index=False))
    lines.append("")

    lines.append("4. PHASE 2 OUTPUT FILES")
    lines.append("-" * 80)
    lines.append(str(PROCESSED_DIR / "clean_earthquakes.csv"))
    lines.append(str(PROCESSED_DIR / "combined_soil_data.csv"))
    lines.append(str(PROCESSED_DIR / "soil_summary_by_province.csv"))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved Phase 2 report to: {report_path}")


def main():
    print("Starting Phase 2: Data Cleaning and Standardization")
    print("=" * 80)

    clean_earthquakes, earthquake_report = clean_earthquake_data()
    print(f"Cleaned earthquake data: {clean_earthquakes.shape}")

    combined_soil, soil_summary, soil_report = clean_and_combine_soil_data()
    print(f"Combined soil data: {combined_soil.shape}")
    print(f"Soil summary: {soil_summary.shape}")

    write_cleaning_report(
        earthquake_report=earthquake_report,
        soil_report=soil_report,
        clean_earthquakes=clean_earthquakes,
        combined_soil=combined_soil,
        soil_summary=soil_summary,
    )

    print("=" * 80)
    print("Phase 2 complete.")
    print(f"Saved: {PROCESSED_DIR / 'clean_earthquakes.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'combined_soil_data.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'soil_summary_by_province.csv'}")


if __name__ == "__main__":
    main()