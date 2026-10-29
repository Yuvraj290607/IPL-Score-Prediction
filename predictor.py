from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ipl_training_data.csv"
MODEL_PATH = ROOT / "ipl_score_model.joblib"



def load_model():
    """Load the trained IPL prediction model."""
    if not MODEL_PATH.exists():
        return None, None

    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["feature_columns"]
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}") from e



def load_training_data():
    """Read the prepared IPL training dataset."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

def predict_score(model, feature_columns, current_score, overs_completed, wickets,
                  current_run_rate, batting_team, bowling_team, venue):
    """Build one input row and return the model's score estimate."""

    row = pd.DataFrame([{
        "current_score": float(current_score),
        "overs_completed": float(overs_completed),
        "wickets_lost": int(wickets),
        "current_run_rate": float(current_run_rate),
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
    }])

    prediction = float(model.predict(row[feature_columns])[0])

    prediction = max(float(current_score), min(prediction, 350.0))
    rounded = int(round(prediction))

    return {
        "predicted_score": rounded,
        "range": (
            max(int(current_score), rounded - 15),
            rounded + 15
        ),
    }
