from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ipl_training_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ipl_score_model.joblib"

NUMERIC_FEATURES = ["current_score", "overs_completed", "wickets_lost", "current_run_rate"]
CATEGORICAL_FEATURES = ["batting_team", "bowling_team", "venue"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "final_score"

def main():
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("numbers", "passthrough", NUMERIC_FEATURES),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print("Model training completed.")
    print(f"Rows used: {len(data)}")
    print(f"Test MAE: {mae:.2f} runs")
    print(f"Test RMSE: {rmse:.2f} runs")
    print(f"Saved model to: {MODEL_PATH}")
    print("\nIMPORTANT: metrics only measure performance on synthetic demo data.")
    print("Replace it with verified historical IPL data and match-based/time-aware testing for a serious evaluation.")

if __name__ == "__main__":
    main()
