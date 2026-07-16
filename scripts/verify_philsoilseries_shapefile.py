from pathlib import Path
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAPEFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shapefiles"
    / "philsoilseries"
    / "PhilSoilSeries.shp"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "philsoilseries_shapefile_verification_report.txt"


def main():
    print("Verifying PhilSoilSeries shapefile...")
    print("=" * 80)

    if not SHAPEFILE_PATH.exists():
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

    soil_gdf = gpd.read_file(SHAPEFILE_PATH)

    report_lines = []

    report_lines.append("PHILSOILSERIES SHAPEFILE VERIFICATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Shapefile path: {SHAPEFILE_PATH}")
    report_lines.append(f"Shape: {soil_gdf.shape}")
    report_lines.append(f"CRS: {soil_gdf.crs}")
    report_lines.append(f"Geometry column: {soil_gdf.geometry.name}")
    report_lines.append(f"Geometry types: {soil_gdf.geometry.geom_type.value_counts().to_string()}")
    report_lines.append("")
    
    report_lines.append("Columns:")
    for col in soil_gdf.columns:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Missing Values:")
    report_lines.append(soil_gdf.isna().sum().to_string())
    report_lines.append("")

    report_lines.append("Duplicate Rows:")
    report_lines.append(str(soil_gdf.duplicated().sum()))
    report_lines.append("")

    report_lines.append("Bounds:")
    report_lines.append(str(soil_gdf.total_bounds))
    report_lines.append("")

    report_lines.append("First 10 Rows:")
    report_lines.append(soil_gdf.head(10).to_string())
    report_lines.append("")

    # Check invalid or empty geometries
    empty_geometries = soil_gdf.geometry.is_empty.sum()
    null_geometries = soil_gdf.geometry.isna().sum()
    invalid_geometries = (~soil_gdf.geometry.is_valid).sum()

    report_lines.append("Geometry Quality:")
    report_lines.append(f"Empty geometries: {empty_geometries}")
    report_lines.append(f"Null geometries: {null_geometries}")
    report_lines.append(f"Invalid geometries: {invalid_geometries}")
    report_lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Loaded shapefile: {soil_gdf.shape}")
    print(f"CRS: {soil_gdf.crs}")
    print(f"Geometry types:")
    print(soil_gdf.geometry.geom_type.value_counts())
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()