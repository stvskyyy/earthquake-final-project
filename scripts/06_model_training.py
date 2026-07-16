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

PERFORMANCE_PATH = TABLES_DIR / "model_performance_summary.csv"
PREDICTIONS_PATH = TABLES_DIR / "phase6_test_set_predictions.csv"
REPORT_PATH = TABLES_DIR / "phase6_model_training_report.txt"

LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
GRADIENT_BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting_model.pkl"

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

CATEGORICAL_FEATURES = [
    "tsunami",
    "magtype",
    "aez",
    "soil_series_grouped",
    "soil_desc_grouped",
    "depth_category",
]

TARGET_COLUMN = "mag"


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with older and newer scikit-learn versions."""
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


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

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = TABLES_DIR / f"{safe_name}_feature_importance.csv"
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
        path = MODELS_DIR / f"{safe_name}.pkl"

    joblib.dump(model, path)
    return path


def main():
    print("Starting Phase 6: Baseline Machine Learning Models")
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
    prediction_df = pd.DataFrame({"actual_mag": y_test.reset_index(drop=True)})

    saved_model_paths = {}
    feature_importance_paths = {}

    for model_name, model in models.items():
        print(f"Training: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        test_predictions = model.predict(X_test)
        prediction_df[f"predicted_mag_{model_name.lower().replace(' ', '_')}"] = test_predictions

        performance_rows.append({"model": model_name, **metrics})

        saved_path = save_model(model_name, model)
        saved_model_paths[model_name] = saved_path

        importance_path = save_feature_importance(model_name, model)
        if importance_path is not None:
            feature_importance_paths[model_name] = importance_path

    performance_df = pd.DataFrame(performance_rows).sort_values("test_rmse", ascending=True)

    performance_df.to_csv(PERFORMANCE_PATH, index=False)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    best_model_row = performance_df.iloc[0]
    best_model_name = best_model_row["model"]

    report_lines = []
    report_lines.append("PHASE 6 BASELINE MACHINE LEARNING MODEL TRAINING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Original dataset shape: {original_shape}")
    report_lines.append(f"Dataset shape after required-column dropna: {df.shape}")
    report_lines.append("")

    report_lines.append("1. MODELING SETUP")
    report_lines.append("-" * 80)
    report_lines.append(f"Target variable: {TARGET_COLUMN}")
    report_lines.append(f"Train-test split: {int((1 - TEST_SIZE) * 100)}% train / {int(TEST_SIZE * 100)}% test")
    report_lines.append(f"Random state: {RANDOM_STATE}")
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

    report_lines.append("2. TRAINED MODELS")
    report_lines.append("-" * 80)
    for model_name in models.keys():
        report_lines.append(f"- {model_name}")
    report_lines.append("")

    report_lines.append("3. MODEL PERFORMANCE")
    report_lines.append("-" * 80)
    report_lines.append(performance_df.to_string(index=False))
    report_lines.append("")
    report_lines.append(f"Saved performance summary to: {PERFORMANCE_PATH}")
    report_lines.append(f"Saved test-set predictions to: {PREDICTIONS_PATH}")
    report_lines.append("")

    report_lines.append("4. BEST BASELINE MODEL")
    report_lines.append("-" * 80)
    report_lines.append(f"Best model by lowest test RMSE: {best_model_name}")
    report_lines.append(f"Test MAE: {best_model_row['test_mae']:.4f}")
    report_lines.append(f"Test RMSE: {best_model_row['test_rmse']:.4f}")
    report_lines.append(f"Test R^2: {best_model_row['test_r2']:.4f}")
    report_lines.append("")

    report_lines.append("5. SAVED MODEL FILES")
    report_lines.append("-" * 80)
    for model_name, path in saved_model_paths.items():
        report_lines.append(f"{model_name}: {path}")
    report_lines.append("")

    report_lines.append("6. SAVED FEATURE IMPORTANCE FILES")
    report_lines.append("-" * 80)
    if feature_importance_paths:
        for model_name, path in feature_importance_paths.items():
            report_lines.append(f"{model_name}: {path}")
    else:
        report_lines.append("No feature importance files generated.")
    report_lines.append("")

    report_lines.append("7. INTERPRETATION GUIDE")
    report_lines.append("-" * 80)
    report_lines.append(
        "MAE measures the average absolute prediction error in magnitude units. "
        "RMSE penalizes larger errors more strongly than MAE. "
        "R^2 estimates how much target variation is explained by the model, where values closer to 1 indicate stronger fit."
    )
    report_lines.append("")
    report_lines.append(
        "A large gap between train performance and test performance may indicate overfitting. "
        "Because earthquake magnitude is complex and the dataset is dominated by light magnitude events, "
        "the baseline models may perform better on common magnitudes than on rare strong or major earthquakes."
    )
    report_lines.append("")

    report_lines.append("8. SCIENTIFIC LIMITATION")
    report_lines.append("-" * 80)
    report_lines.append(
        "The models estimate earthquake magnitude using seismic, spatial, temporal, and soil-context features. "
        "Soil variables should be interpreted as contextual geospatial predictors, not direct causal determinants of earthquake magnitude. "
        "Soil conditions are more directly related to site response, ground shaking amplification, and intensity."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=" * 80)
    print("Phase 6 complete.")
    print(f"Performance summary: {PERFORMANCE_PATH}")
    print(f"Test-set predictions: {PREDICTIONS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Best model by test RMSE: {best_model_name}")


if __name__ == "__main__":
    main()
