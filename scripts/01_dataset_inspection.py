from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def inspect_csv_files():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files.")

    inventory = []
    report_lines = []

    report_lines.append("PHASE 1 DATASET INSPECTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Raw data folder: {RAW_DATA_DIR}")
    report_lines.append(f"Total CSV files loaded: {len(csv_files)}")
    report_lines.append("")

    for file in csv_files:
        print(f"Reading: {file.name}")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"ERROR reading {file.name}: {e}")
            continue

        missing_total = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        inventory.append({
            "dataset_name": file.stem,
            "file_name": file.name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values_total": missing_total,
            "duplicate_rows": duplicate_rows,
            "column_names": ", ".join(df.columns.astype(str))
        })

        report_lines.append(f"DATASET: {file.stem}")
        report_lines.append(f"File name: {file.name}")
        report_lines.append(f"Shape: {df.shape}")
        report_lines.append("")

        report_lines.append("Columns:")
        for col in df.columns:
            report_lines.append(f"- {col}")

        report_lines.append("")
        report_lines.append("Data Types:")
        report_lines.append(df.dtypes.to_string())

        report_lines.append("")
        report_lines.append("Missing Values:")
        report_lines.append(df.isna().sum().to_string())

        report_lines.append("")
        report_lines.append("Duplicate Rows:")
        report_lines.append(str(duplicate_rows))

        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("")

    inventory_df = pd.DataFrame(inventory)

    inventory_path = OUTPUT_TABLES_DIR / "dataset_inventory.csv"
    report_path = OUTPUT_TABLES_DIR / "phase1_dataset_inspection_report.txt"

    inventory_df.to_csv(inventory_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nDone.")
    print(f"Saved inventory to: {inventory_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    inspect_csv_files()