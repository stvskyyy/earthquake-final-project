from pathlib import Path
import pandas as pd
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SHAPEFILE_DIR = PROJECT_ROOT / "data" / "raw" / "shapefiles" / "philsoilseries"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

EARTHQUAKE_PATH = PROCESSED_DIR / "clean_earthquakes.csv"
SOIL_SHAPEFILE_PATH = SHAPEFILE_DIR / "PhilSoilSeries.shp"

OUTPUT_ALL_PATH = PROCESSED_DIR / "earthquakes_with_soil_all.csv"
OUTPUT_MATCHED_PATH = PROCESSED_DIR / "model_dataset_spatial.csv"
REPORT_PATH = OUTPUT_TABLES_DIR / "phase3_spatial_integration_report.txt"


def clean_soil_columns(soil_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    soil_gdf = soil_gdf.copy()

    soil_gdf = soil_gdf.rename(
        columns={
            "SOILDESC": "soil_desc",
            "AEZ": "aez",
            "SoilSeries": "soil_series",
            "Hectares": "soil_polygon_hectares",
        }
    )

    text_cols = ["soil_desc", "aez", "soil_series"]

    for col in text_cols:
        if col in soil_gdf.columns:
            soil_gdf[col] = (
                soil_gdf[col]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    return soil_gdf


def load_earthquake_points() -> gpd.GeoDataFrame:
    if not EARTHQUAKE_PATH.exists():
        raise FileNotFoundError(f"Missing earthquake file: {EARTHQUAKE_PATH}")

    eq = pd.read_csv(EARTHQUAKE_PATH)

    required_cols = ["longitude", "latitude", "mag", "depth", "time"]

    missing_required = [col for col in required_cols if col not in eq.columns]
    if missing_required:
        raise ValueError(f"Earthquake file is missing required columns: {missing_required}")

    eq_gdf = gpd.GeoDataFrame(
        eq,
        geometry=gpd.points_from_xy(eq["longitude"], eq["latitude"]),
        crs="EPSG:4326",
    )

    return eq_gdf


def load_soil_polygons() -> gpd.GeoDataFrame:
    if not SOIL_SHAPEFILE_PATH.exists():
        raise FileNotFoundError(f"Missing soil shapefile: {SOIL_SHAPEFILE_PATH}")

    soil_gdf = gpd.read_file(SOIL_SHAPEFILE_PATH)
    soil_gdf = clean_soil_columns(soil_gdf)

    if soil_gdf.crs is None:
        raise ValueError("Soil shapefile has no CRS. Cannot safely perform spatial join.")

    soil_gdf = soil_gdf.to_crs("EPSG:4326")

    soil_gdf = soil_gdf[
        [
            "soil_desc",
            "aez",
            "soil_series",
            "soil_polygon_hectares",
            "geometry",
        ]
    ].copy()

    return soil_gdf


def create_spatial_join(eq_gdf: gpd.GeoDataFrame, soil_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        eq_gdf,
        soil_gdf,
        how="left",
        predicate="within",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined


def prepare_model_dataset(joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    df = joined_gdf.copy()

    # Remove geometry for CSV modeling dataset.
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    # Keep only earthquake records that matched a soil polygon.
    matched = df.dropna(subset=["soil_desc"]).copy()

    # Exclude leakage columns from the modeling-ready file.
    # title contains magnitude text, magnitude_category is derived from mag,
    # and sig is strongly related to event significance/magnitude.
    leakage_cols = ["title", "magnitude_category", "sig"]

    matched = matched.drop(columns=[col for col in leakage_cols if col in matched.columns], errors="ignore")

    return matched


def write_report(
    eq_gdf: gpd.GeoDataFrame,
    soil_gdf: gpd.GeoDataFrame,
    joined_gdf: gpd.GeoDataFrame,
    model_df: pd.DataFrame,
) -> None:
    total_eq = len(eq_gdf)
    matched_count = joined_gdf["soil_desc"].notna().sum()
    unmatched_count = joined_gdf["soil_desc"].isna().sum()

    match_rate = matched_count / total_eq * 100 if total_eq > 0 else 0

    lines = []

    lines.append("PHASE 3 SPATIAL DATA INTEGRATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. INPUT DATA")
    lines.append("-" * 80)
    lines.append(f"Earthquake file: {EARTHQUAKE_PATH}")
    lines.append(f"Soil shapefile: {SOIL_SHAPEFILE_PATH}")
    lines.append("")
    lines.append(f"Earthquake records loaded: {eq_gdf.shape}")
    lines.append(f"Soil polygon records loaded: {soil_gdf.shape}")
    lines.append(f"Soil CRS after conversion: {soil_gdf.crs}")
    lines.append("")

    lines.append("2. SPATIAL JOIN RESULT")
    lines.append("-" * 80)
    lines.append(f"Total earthquake records: {total_eq}")
    lines.append(f"Earthquake records matched to soil polygon: {matched_count}")
    lines.append(f"Earthquake records not matched to soil polygon: {unmatched_count}")
    lines.append(f"Match rate: {match_rate:.2f}%")
    lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "Matched records are earthquake points located within soil polygons. "
        "Unmatched records are likely offshore events, events outside the soil shapefile coverage, "
        "or records located in areas classified as missing/no soil polygon coverage."
    )
    lines.append("")

    lines.append("3. MODEL DATASET")
    lines.append("-" * 80)
    lines.append(f"Model dataset shape: {model_df.shape}")
    lines.append(f"Saved all joined records to: {OUTPUT_ALL_PATH}")
    lines.append(f"Saved matched model dataset to: {OUTPUT_MATCHED_PATH}")
    lines.append("")

    lines.append("Model dataset columns:")
    for col in model_df.columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("Missing values in model dataset:")
    lines.append(model_df.isna().sum().to_string())
    lines.append("")

    lines.append("Top soil descriptions among matched earthquake records:")
    if "soil_desc" in model_df.columns and len(model_df) > 0:
        lines.append(model_df["soil_desc"].value_counts(dropna=False).head(20).to_string())
    else:
        lines.append("No matched soil descriptions found.")
    lines.append("")

    lines.append("Top soil series among matched earthquake records:")
    if "soil_series" in model_df.columns and len(model_df) > 0:
        lines.append(model_df["soil_series"].value_counts(dropna=False).head(20).to_string())
    else:
        lines.append("No matched soil series found.")
    lines.append("")

    lines.append("Magnitude summary for matched model dataset:")
    if "mag" in model_df.columns and len(model_df) > 0:
        lines.append(model_df["mag"].describe().to_string())
    else:
        lines.append("No magnitude data found.")
    lines.append("")

    lines.append("Depth summary for matched model dataset:")
    if "depth" in model_df.columns and len(model_df) > 0:
        lines.append(model_df["depth"].describe().to_string())
    else:
        lines.append("No depth data found.")
    lines.append("")

    lines.append("4. RESEARCH LIMITATION")
    lines.append("-" * 80)
    lines.append(
        "The spatial join uses earthquake epicenter coordinates and surface soil polygons. "
        "This allows the study to associate earthquake records with the soil polygon at the epicentral location. "
        "However, many Philippine earthquakes occur offshore, so not all earthquake records are expected to match land-based soil polygons."
    )
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Starting Phase 3: Spatial Data Integration")
    print("=" * 80)

    eq_gdf = load_earthquake_points()
    print(f"Loaded earthquake points: {eq_gdf.shape}")

    soil_gdf = load_soil_polygons()
    print(f"Loaded soil polygons: {soil_gdf.shape}")
    print(f"Soil CRS after conversion: {soil_gdf.crs}")

    joined_gdf = create_spatial_join(eq_gdf, soil_gdf)
    print(f"Spatial join complete: {joined_gdf.shape}")

    all_joined_df = joined_gdf.drop(columns=["geometry"], errors="ignore")
    all_joined_df.to_csv(OUTPUT_ALL_PATH, index=False)

    model_df = prepare_model_dataset(joined_gdf)
    model_df.to_csv(OUTPUT_MATCHED_PATH, index=False)

    write_report(eq_gdf, soil_gdf, joined_gdf, model_df)

    matched_count = joined_gdf["soil_desc"].notna().sum()
    unmatched_count = joined_gdf["soil_desc"].isna().sum()

    print("=" * 80)
    print("Phase 3 complete.")
    print(f"Matched earthquake records: {matched_count}")
    print(f"Unmatched earthquake records: {unmatched_count}")
    print(f"Saved all joined records to: {OUTPUT_ALL_PATH}")
    print(f"Saved model dataset to: {OUTPUT_MATCHED_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()