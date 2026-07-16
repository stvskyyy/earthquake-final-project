from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "model_dataset_features.csv"

PERFORMANCE_PATH = TABLES_DIR / "model_performance_summary_no_magtype.csv"
PREDICTIONS_PATH = TABLES_DIR / "phase6b_test_set_predictions_no_magtype.csv"
REPORT_PATH = TABLES_DIR / "phase6b_model_training_without_magtype_report.txt"

LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression_model_no_magtype.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model_no_magtype.pkl"
GRADIENT_BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting_model_no_magtype.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


NUMERICAL_FEATURES = [
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

# Deliberately excludes magtype.
CATEGORICAL_FEATURES = [
    "tsunami",
    "aez",
    "soil_series_grouped",
    "soil_desc_grouped",
    "depth_category",
]

EXCLUDED_FOR_ROBUSTNESS_TEST = [
    "magtype",
]

TARGET_COLUMN = "mag"


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Creates a OneHotEncoder that works across recent and older scikit-learn versions.
    Newer versions use sparse_output; older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_models() -> dict:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
    }


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_names = NUMERICAL_FEATURES

    categorical_pipeline = preprocessor.named_transformers_["cat"]
    onehot = categorical_pipeline.named_steps["onehot"]
    categorical_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    return numeric_names + categorical_names


def save_feature_importance(model_name: str, pipeline: Pipeline) -> Path | None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = get_feature_names_from_pipeline(pipeline)
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_path = TABLES_DIR / f"{safe_name}_feature_importance_no_magtype.csv"
    importance_df.to_csv(output_path, index=False)

    return output_path


def save_model(model_name: str, model: Pipeline) -> Path:
    if model_name == "Linear Regression":
        path = LINEAR_MODEL_PATH
    elif model_name == "Random Forest Regressor":
        path = RANDOM_FOREST_MODEL_PATH
    elif model_name == "Gradient Boosting Regressor":
        path = GRADIENT_BOOSTING_MODEL_PATH
    else:
        safe_name = model_name.lower().replace(" ", "_")
        path = MODELS_DIR / f"{safe_name}_no_magtype.pkl"

    joblib.dump(model, path)
    return path


def create_error_by_magnitude_category(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    error_df = pd.DataFrame(
        {
            "actual_mag": y_true.reset_index(drop=True),
            "predicted_mag": y_pred,
        }
    )

    error_df["absolute_error"] = (error_df["actual_mag"] - error_df["predicted_mag"]).abs()

    error_df["magnitude_category"] = pd.cut(
        error_df["actual_mag"],
        bins=[0, 3.9, 4.9, 5.9, 6.9, 10],
        labels=[
            "Minor (<4.0)",
            "Light (4.0-4.9)",
            "Moderate (5.0-5.9)",
            "Strong (6.0-6.9)",
            "Major (7.0+)",
        ],
        include_lowest=True,
    )

    return (
        error_df
        .groupby("magnitude_category", observed=False)["absolute_error"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )


def main():
    print("Starting Phase 6B: Robustness Model Training Without magtype")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    original_shape = df.shape

    required_columns = [TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    performance_rows = []
    prediction_df = pd.DataFrame(
        {
            "actual_mag": y_test.reset_index(drop=True),
        }
    )

    saved_model_paths = {}
    feature_importance_paths = {}
    best_predictions = None

    for model_name, model in models.items():
        print(f"Training: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        test_predictions = model.predict(X_test)
        prediction_df[f"predicted_mag_{model_name.lower().replace(' ', '_')}"] = test_predictions

        row = {
            "model": model_name,
            **metrics,
        }
        performance_rows.append(row)

        saved_path = save_model(model_name, model)
        saved_model_paths[model_name] = saved_path

        importance_path = save_feature_importance(model_name, model)
        if importance_path is not None:
            feature_importance_paths[model_name] = importance_path

    performance_df = pd.DataFrame(performance_rows)
    performance_df = performance_df.sort_values("test_rmse", ascending=True)

    best_model_row = performance_df.iloc[0]
    best_model_name = best_model_row["model"]

    best_prediction_col = f"predicted_mag_{best_model_name.lower().replace(' ', '_')}"
    best_predictions = prediction_df[best_prediction_col].to_numpy()

    error_by_magnitude = create_error_by_magnitude_category(y_test, best_predictions)
    error_by_magnitude_path = TABLES_DIR / "phase6b_error_by_magnitude_category_no_magtype.csv"
    error_by_magnitude.to_csv(error_by_magnitude_path, index=False)

    performance_df.to_csv(PERFORMANCE_PATH, index=False)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    report_lines = []
    report_lines.append("PHASE 6B ROBUSTNESS MODEL TRAINING WITHOUT MAGTYPE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Original dataset shape: {original_shape}")
    report_lines.append(f"Dataset shape after required-column dropna: {df.shape}")
    report_lines.append("")

    report_lines.append("1. PURPOSE OF THIS EXPERIMENT")
    report_lines.append("-" * 80)
    report_lines.append(
        "This experiment repeats the baseline regression modeling workflow while excluding magtype. "
        "The goal is to test whether model performance depends heavily on the magnitude measurement type. "
        "This is important because magtype may not be available in a real-world pre-measurement prediction setting."
    )
    report_lines.append("")

    report_lines.append("2. MODELING SETUP")
    report_lines.append("-" * 80)
    report_lines.append(f"Target variable: {TARGET_COLUMN}")
    report_lines.append(f"Train-test split: {int((1 - TEST_SIZE) * 100)}% train / {int(TEST_SIZE * 100)}% test")
    report_lines.append(f"Random state: {RANDOM_STATE}")
    report_lines.append("")

    report_lines.append("Excluded feature for robustness test:")
    for col in EXCLUDED_FOR_ROBUSTNESS_TEST:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Numerical features:")
    for col in NUMERICAL_FEATURES:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Categorical features:")
    for col in CATEGORICAL_FEATURES:
        report_lines.append(f"- {col}")
    report_lines.append("")

    report_lines.append("Preprocessing:")
    report_lines.append("- Numerical features: median imputation + standard scaling")
    report_lines.append("- Categorical features: most-frequent imputation + one-hot encoding")
    report_lines.append("")

    report_lines.append("3. TRAINED MODELS")
    report_lines.append("-" * 80)
    for model_name in models.keys():
        report_lines.append(f"- {model_name}")
    report_lines.append("")

    report_lines.append("4. MODEL PERFORMANCE WITHOUT MAGTYPE")
    report_lines.append("-" * 80)
    report_lines.append(performance_df.to_string(index=False))
    report_lines.append("")
    report_lines.append(f"Saved performance summary to: {PERFORMANCE_PATH}")
    report_lines.append(f"Saved test-set predictions to: {PREDICTIONS_PATH}")
    report_lines.append("")

    report_lines.append("5. BEST MODEL WITHOUT MAGTYPE")
    report_lines.append("-" * 80)
    report_lines.append(f"Best model by lowest test RMSE: {best_model_name}")
    report_lines.append(f"Test MAE: {best_model_row['test_mae']:.4f}")
    report_lines.append(f"Test RMSE: {best_model_row['test_rmse']:.4f}")
    report_lines.append(f"Test R²: {best_model_row['test_r2']:.4f}")
    report_lines.append("")

    report_lines.append("6. ERROR BY MAGNITUDE CATEGORY FOR BEST MODEL")
    report_lines.append("-" * 80)
    report_lines.append(error_by_magnitude.to_string(index=False))
    report_lines.append("")
    report_lines.append(f"Saved error-by-category summary to: {error_by_magnitude_path}")
    report_lines.append("")

    report_lines.append("7. SAVED MODEL FILES")
    report_lines.append("-" * 80)
    for model_name, path in saved_model_paths.items():
        report_lines.append(f"{model_name}: {path}")
    report_lines.append("")

    report_lines.append("8. SAVED FEATURE IMPORTANCE FILES")
    report_lines.append("-" * 80)
    if feature_importance_paths:
        for model_name, path in feature_importance_paths.items():
            report_lines.append(f"{model_name}: {path}")
    else:
        report_lines.append("No feature importance files generated.")
    report_lines.append("")

    report_lines.append("9. INTERPRETATION GUIDE")
    report_lines.append("-" * 80)
    report_lines.append(
        "Compare these results with Phase 6. If test performance drops substantially after removing magtype, "
        "then the original model depended heavily on the magnitude-type feature. If performance remains similar, "
        "then spatial, temporal, depth, and soil-context features still provide predictive information without magtype."
    )
    report_lines.append("")

    report_lines.append("10. SCIENTIFIC LIMITATION")
    report_lines.append("-" * 80)
    report_lines.append(
        "This no-magtype model is more conservative for interpretation because it avoids relying on a catalog-specific magnitude measurement label. "
        "However, the model still estimates magnitude from observational and contextual features, and should not be interpreted as a deterministic earthquake prediction system."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=" * 80)
    print("Phase 6B complete.")
    print(f"Performance summary: {PERFORMANCE_PATH}")
    print(f"Test-set predictions: {PREDICTIONS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Error by magnitude category: {error_by_magnitude_path}")
    print(f"Best model by test RMSE: {best_model_name}")


if __name__ == "__main__":
    main()
