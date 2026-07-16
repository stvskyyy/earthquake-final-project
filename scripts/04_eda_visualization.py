from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "model_dataset_spatial_clean.csv"
REPORT_PATH = TABLES_DIR / "phase4_eda_report.txt"


def save_magnitude_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df["mag"], bins=30, edgecolor="black")
    plt.title("Distribution of Earthquake Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "magnitude_distribution.png", dpi=300)
    plt.close()


def save_depth_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df["depth"], bins=30, edgecolor="black")
    plt.title("Distribution of Earthquake Depths")
    plt.xlabel("Depth (km)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "depth_distribution.png", dpi=300)
    plt.close()


def save_depth_vs_magnitude(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["depth"], df["mag"], alpha=0.5)
    plt.title("Depth vs Earthquake Magnitude")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "depth_vs_magnitude.png", dpi=300)
    plt.close()


def save_earthquake_count_by_aez(df):
    counts = df["aez"].value_counts()

    plt.figure(figsize=(8, 6))
    counts.plot(kind="bar")
    plt.title("Earthquake Count by Agro-Ecological Zone")
    plt.xlabel("AEZ")
    plt.ylabel("Number of Earthquake Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "earthquake_count_by_aez.png", dpi=300)
    plt.close()


def save_magnitude_by_aez(df):
    aez_order = (
        df.groupby("aez")["mag"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [
        df.loc[df["aez"] == aez, "mag"].dropna().astype(float).tolist()
        for aez in aez_order
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, tick_labels=aez_order)
    plt.title("Earthquake Magnitude by Agro-Ecological Zone")
    plt.xlabel("AEZ")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "magnitude_by_aez.png", dpi=300)
    plt.close()

def save_top_soil_series_counts(df, top_n=15):
    counts = df["soil_series"].value_counts().head(top_n)

    plt.figure(figsize=(12, 7))
    counts.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Soil Series by Earthquake Record Count")
    plt.xlabel("Number of Earthquake Records")
    plt.ylabel("Soil Series")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_soil_series_counts.png", dpi=300)
    plt.close()


def save_mean_magnitude_by_soil_series(df, top_n=15):
    top_soils = df["soil_series"].value_counts().head(top_n).index
    subset = df[df["soil_series"].isin(top_soils)]

    mean_mag = subset.groupby("soil_series")["mag"].mean().sort_values()

    plt.figure(figsize=(12, 7))
    mean_mag.plot(kind="barh")
    plt.title(f"Mean Magnitude by Top {top_n} Soil Series")
    plt.xlabel("Mean Magnitude")
    plt.ylabel("Soil Series")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_magnitude_by_top_soil_series.png", dpi=300)
    plt.close()


def save_yearly_earthquake_counts(df):
    yearly_counts = df["year"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind="line", marker="o")
    plt.title("Yearly Count of Matched Earthquake Records")
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquake Records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "yearly_earthquake_counts.png", dpi=300)
    plt.close()


def save_monthly_earthquake_counts(df):
    monthly_counts = df["month"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    monthly_counts.plot(kind="bar")
    plt.title("Earthquake Record Count by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Earthquake Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_earthquake_counts.png", dpi=300)
    plt.close()


def save_spatial_scatter(df):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["mag"],
        alpha=0.6,
        s=15,
    )
    plt.colorbar(scatter, label="Magnitude")
    plt.title("Spatial Distribution of Matched Earthquake Records")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spatial_distribution_magnitude.png", dpi=300)
    plt.close()


def write_eda_report(df):
    lines = []

    lines.append("PHASE 4 EXPLORATORY DATA ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Input file: {INPUT_PATH}")
    lines.append(f"Dataset shape: {df.shape}")
    lines.append("")

    lines.append("1. DATASET COLUMNS")
    lines.append("-" * 80)
    for col in df.columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("2. MISSING VALUES")
    lines.append("-" * 80)
    lines.append(df.isna().sum().to_string())
    lines.append("")

    lines.append("3. TARGET VARIABLE: MAGNITUDE")
    lines.append("-" * 80)
    lines.append(df["mag"].describe().to_string())
    lines.append("")

    lines.append("Magnitude category counts:")
    mag_bins = pd.cut(
        df["mag"],
        bins=[0, 3.9, 4.9, 5.9, 6.9, 10],
        labels=["Minor (<4.0)", "Light (4.0-4.9)", "Moderate (5.0-5.9)", "Strong (6.0-6.9)", "Major (7.0+)"],
    )
    lines.append(mag_bins.value_counts().sort_index().to_string())
    lines.append("")

    lines.append("4. DEPTH")
    lines.append("-" * 80)
    lines.append(df["depth"].describe().to_string())
    lines.append("")

    lines.append("Depth category counts:")
    depth_bins = pd.cut(
        df["depth"],
        bins=[0, 70, 300, 700],
        labels=["Shallow (0-70 km)", "Intermediate (70-300 km)", "Deep (300-700 km)"],
    )
    lines.append(depth_bins.value_counts().sort_index().to_string())
    lines.append("")

    lines.append("5. AEZ SUMMARY")
    lines.append("-" * 80)
    lines.append("AEZ counts:")
    lines.append(df["aez"].value_counts().to_string())
    lines.append("")

    lines.append("Magnitude by AEZ:")
    lines.append(df.groupby("aez")["mag"].agg(["count", "mean", "median", "std", "min", "max"]).to_string())
    lines.append("")

    lines.append("Depth by AEZ:")
    lines.append(df.groupby("aez")["depth"].agg(["count", "mean", "median", "std", "min", "max"]).to_string())
    lines.append("")

    lines.append("6. SOIL SERIES SUMMARY")
    lines.append("-" * 80)
    lines.append("Top 20 soil series by record count:")
    lines.append(df["soil_series"].value_counts().head(20).to_string())
    lines.append("")

    lines.append("Top 20 soil descriptions by record count:")
    lines.append(df["soil_desc"].value_counts().head(20).to_string())
    lines.append("")

    top_soils = df["soil_series"].value_counts().head(20).index
    soil_summary = (
        df[df["soil_series"].isin(top_soils)]
        .groupby("soil_series")["mag"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )

    lines.append("Magnitude summary for top 20 soil series:")
    lines.append(soil_summary.to_string())
    lines.append("")

    lines.append("7. TEMPORAL SUMMARY")
    lines.append("-" * 80)
    lines.append("Yearly earthquake record counts:")
    lines.append(df["year"].value_counts().sort_index().to_string())
    lines.append("")

    lines.append("Monthly earthquake record counts:")
    lines.append(df["month"].value_counts().sort_index().to_string())
    lines.append("")

    lines.append("8. CORRELATION SUMMARY")
    lines.append("-" * 80)
    numeric_cols = [
        "mag",
        "longitude",
        "latitude",
        "depth",
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "soil_polygon_hectares",
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr = df[numeric_cols].corr(numeric_only=True)

    lines.append("Correlation with magnitude:")
    lines.append(corr["mag"].sort_values(ascending=False).to_string())
    lines.append("")

    lines.append("9. GENERATED FIGURES")
    lines.append("-" * 80)
    generated_figures = [
        "magnitude_distribution.png",
        "depth_distribution.png",
        "depth_vs_magnitude.png",
        "earthquake_count_by_aez.png",
        "magnitude_by_aez.png",
        "top_soil_series_counts.png",
        "mean_magnitude_by_top_soil_series.png",
        "yearly_earthquake_counts.png",
        "monthly_earthquake_counts.png",
        "spatial_distribution_magnitude.png",
    ]

    for fig in generated_figures:
        lines.append(f"- outputs/figures/{fig}")

    lines.append("")
    lines.append("10. INITIAL INTERPRETATION NOTES")
    lines.append("-" * 80)
    lines.append(
        "This EDA focuses on the 4,083 land-matched earthquake records with assigned soil attributes. "
        "The analysis should be interpreted within the scope of land-based earthquake epicenters only. "
        "Offshore events were excluded because they could not be assigned surface soil polygon attributes."
    )
    lines.append("")
    lines.append(
        "Soil variables such as soil_desc, soil_series, and aez are contextual spatial predictors. "
        "They should not be interpreted as direct causal factors of earthquake magnitude. "
        "They are more directly relevant to ground shaking, amplification, and site effects."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Starting Phase 4: Exploratory Data Analysis")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    print(f"Loaded dataset: {df.shape}")

    save_magnitude_distribution(df)
    save_depth_distribution(df)
    save_depth_vs_magnitude(df)
    save_earthquake_count_by_aez(df)
    save_magnitude_by_aez(df)
    save_top_soil_series_counts(df)
    save_mean_magnitude_by_soil_series(df)
    save_yearly_earthquake_counts(df)
    save_monthly_earthquake_counts(df)
    save_spatial_scatter(df)

    write_eda_report(df)

    print("Phase 4 complete.")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved EDA report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()