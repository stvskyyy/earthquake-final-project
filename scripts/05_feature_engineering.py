from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "model_dataset_spatial_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "model_dataset_features.csv"
REPORT_PATH = TABLES_DIR / "phase5_feature_engineering_report.txt"


RARE_CATEGORY_THRESHOLD = 30


def group_rare_categories(series: pd.Series, threshold: int = RARE_CATEGORY_THRESHOLD) -> pd.Series:
    """
    Groups rare categorical values into 'Other'.

    Example:
    If a soil series appears fewer than 30 times, it becomes 'Other'.
    """
    counts = series.value_counts(dropna=False)
    common_categories = counts[counts >= threshold].index

    return series.where(series.isin(common_categories), "Other")


def create_depth_category(depth: pd.Series) -> pd.Series:
    """
    Standard earthquake depth classification:
    - Shallow: 0 to 70 km
    - Intermediate: 70 to 300 km
    - Deep: 300 to 700 km
    """
    return pd.cut(
        depth,
        bins=[0, 70, 300, 700],
        labels=[
            "Shallow",
            "Intermediate",
            "Deep",
        ],
        include_lowest=True,
    ).astype("string")


def create_magnitude_category(mag: pd.Series) -> pd.Series:
    """
    For analysis only. This must NOT be used as a model predictor
    because it is derived from the target variable.
    """
    return pd.cut(
        mag,
        bins=[0, 3.9, 4.9, 5.9, 6.9, 10],
        labels=[
            "Minor",
            "Light",
            "Moderate",
            "Strong",
            "Major",
        ],
        include_lowest=True,
    ).astype("string")


def main():
    print("Starting Phase 5: Feature Engineering")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    original_shape = df.shape

    report_lines = []
    report_lines.append("PHASE 5 FEATURE ENGINEERING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Original shape: {original_shape}")
    report_lines.append("")

    # ------------------------------------------------------------
    # 1. Basic type cleaning
    # ------------------------------------------------------------
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    report_lines.append("1. BASIC TYPE CLEANING")
    report_lines.append("-" * 80)
    report_lines.append("Converted time column to datetime.")
    report_lines.append("Cleaned string spacing for categorical columns.")
    report_lines.append("")

    # ------------------------------------------------------------
    # 2. Rare category grouping
    # ------------------------------------------------------------
    if "soil_series" in df.columns:
        df["soil_series_grouped"] = group_rare_categories(df["soil_series"])

    if "soil_desc" in df.columns:
        df["soil_desc_grouped"] = group_rare_categories(df["soil_desc"])

    report_lines.append("2. RARE CATEGORY GROUPING")
    report_lines.append("-" * 80)
    report_lines.append(f"Rare category threshold: fewer than {RARE_CATEGORY_THRESHOLD} records")
    report_lines.append("")

    if "soil_series_grouped" in df.columns:
        report_lines.append("Original soil_series unique count:")
        report_lines.append(str(df["soil_series"].nunique()))
        report_lines.append("Grouped soil_series unique count:")
        report_lines.append(str(df["soil_series_grouped"].nunique()))
        report_lines.append("")
        report_lines.append("Grouped soil_series counts:")
        report_lines.append(df["soil_series_grouped"].value_counts().to_string())
        report_lines.append("")

    if "soil_desc_grouped" in df.columns:
        report_lines.append("Original soil_desc unique count:")
        report_lines.append(str(df["soil_desc"].nunique()))
        report_lines.append("Grouped soil_desc unique count:")
        report_lines.append(str(df["soil_desc_grouped"].nunique()))
        report_lines.append("")
        report_lines.append("Grouped soil_desc counts:")
        report_lines.append(df["soil_desc_grouped"].value_counts().to_string())
        report_lines.append("")

    # ------------------------------------------------------------
    # 3. Depth and magnitude categories
    # ------------------------------------------------------------
    if "depth" in df.columns:
        df["depth_category"] = create_depth_category(df["depth"])

    if "mag" in df.columns:
        df["magnitude_category_analysis"] = create_magnitude_category(df["mag"])

    report_lines.append("3. EARTHQUAKE CATEGORY FEATURES")
    report_lines.append("-" * 80)

    if "depth_category" in df.columns:
        report_lines.append("Depth category counts:")
        report_lines.append(df["depth_category"].value_counts().to_string())
        report_lines.append("")

    if "magnitude_category_analysis" in df.columns:
        report_lines.append("Magnitude category counts for analysis only:")
        report_lines.append(df["magnitude_category_analysis"].value_counts().to_string())
        report_lines.append("")
        report_lines.append(
            "NOTE: magnitude_category_analysis is derived from mag and must not be used as a model predictor."
        )
        report_lines.append("")

    # ------------------------------------------------------------
    # 4. Cyclical time features
    # ------------------------------------------------------------
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "day_of_week" in df.columns:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    report_lines.append("4. CYCLICAL TIME FEATURES")
    report_lines.append("-" * 80)
    report_lines.append("Created cyclical features:")
    report_lines.append("- month_sin")
    report_lines.append("- month_cos")
    report_lines.append("- hour_sin")
    report_lines.append("- hour_cos")
    report_lines.append("- day_of_week_sin")
    report_lines.append("- day_of_week_cos")
    report_lines.append("")

    # ------------------------------------------------------------
    # 5. Soil polygon area transformation
    # ------------------------------------------------------------
    if "soil_polygon_hectares" in df.columns:
        df["log_soil_polygon_hectares"] = np.log1p(df["soil_polygon_hectares"])

    report_lines.append("5. SOIL POLYGON AREA TRANSFORMATION")
    report_lines.append("-" * 80)
    report_lines.append("Created log_soil_polygon_hectares using log1p transformation.")
    report_lines.append("This reduces skewness in the soil polygon area variable.")
    report_lines.append("")

    # ------------------------------------------------------------
    # 6. Modeling feature selection
    # ------------------------------------------------------------
    target_column = "mag"

    numerical_features = [
        "longitude",
        "latitude",
        "depth",
        "log_soil_polygon_hectares",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day",
    ]

    categorical_features = [
        "tsunami",
        "magtype",
        "aez",
        "soil_series_grouped",
        "soil_desc_grouped",
        "depth_category",
    ]

    optional_features_not_used_initially = [
        "year",
        "month",
        "hour",
        "day_of_week",
        "soil_series",
        "soil_desc",
        "soil_polygon_hectares",
        "magnitude_category_analysis",
        "time",
    ]

    selected_columns = [target_column] + numerical_features + categorical_features

    existing_selected_columns = [col for col in selected_columns if col in df.columns]
    missing_selected_columns = [col for col in selected_columns if col not in df.columns]

    feature_df = df[existing_selected_columns].copy()

    # Remove rows that somehow became incomplete after feature engineering.
    before_dropna = len(feature_df)
    feature_df = feature_df.dropna().copy()
    after_dropna = len(feature_df)

    # Save feature-engineered dataset.
    feature_df.to_csv(OUTPUT_PATH, index=False)

    report_lines.append("6. MODELING FEATURE SELECTION")
    report_lines.append("-" * 80)
    report_lines.append(f"Target column: {target_column}")
    report_lines.append("")

    report_lines.append("Numerical features:")
    for col in numerical_features:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Categorical features:")
    for col in categorical_features:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Columns intentionally not used in the initial model:")
    for col in optional_features_not_used_initially:
        report_lines.append(f"- {col}")
    report_lines.append("")

    if missing_selected_columns:
        report_lines.append("WARNING: Selected columns missing from dataset:")
        for col in missing_selected_columns:
            report_lines.append(f"- {col}")
        report_lines.append("")

    report_lines.append(f"Rows before final dropna: {before_dropna}")
    report_lines.append(f"Rows after final dropna: {after_dropna}")
    report_lines.append(f"Rows removed after feature engineering: {before_dropna - after_dropna}")
    report_lines.append("")

    report_lines.append("7. FINAL FEATURE DATASET")
    report_lines.append("-" * 80)
    report_lines.append(f"Saved feature dataset to: {OUTPUT_PATH}")
    report_lines.append(f"Final feature dataset shape: {feature_df.shape}")
    report_lines.append("")

    report_lines.append("Final feature dataset columns:")
    for col in feature_df.columns:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Missing values in final feature dataset:")
    report_lines.append(feature_df.isna().sum().to_string())
    report_lines.append("")

    report_lines.append("Target summary:")
    report_lines.append(feature_df["mag"].describe().to_string())
    report_lines.append("")

    report_lines.append("8. METHODOLOGICAL JUSTIFICATION")
    report_lines.append("-" * 80)
    report_lines.append(
        "The engineered dataset combines seismic, spatial, temporal, and soil-related variables. "
        "Latitude, longitude, depth, and time-derived variables are retained because prior earthquake machine learning studies commonly use spatiotemporal and seismic parameters for magnitude or intensity prediction. "
        "Soil-related variables are included as contextual geospatial predictors because local soil conditions are associated with ground response, shaking amplification, and site effects."
    )
    report_lines.append("")
    report_lines.append(
        "Rare soil categories were grouped into 'Other' to reduce categorical sparsity and improve model stability. "
        "Cyclical transformations were used for month, hour, and day_of_week because these variables are periodic. "
        "The magnitude category feature was kept only for analysis and excluded from modeling because it is derived from the target variable."
    )
    report_lines.append("")
    report_lines.append(
        "The variable year was excluded from the initial feature set because the EDA showed a strong negative relationship with magnitude that may reflect historical catalog/reporting bias rather than a physical relationship."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Phase 5 complete.")
    print(f"Original dataset shape: {original_shape}")
    print(f"Final feature dataset shape: {feature_df.shape}")
    print(f"Saved feature dataset to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()