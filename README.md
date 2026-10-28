# IPL Score Prediction System (Beginner Full-Stack ML Project)

A small educational application:
- Frontend: Streamlit web interface
- Backend: Python prediction functions and a saved Scikit-learn model
- ML: Random Forest Regression
- Data: CSV

## Important data note
`data/ipl_demo_data.csv` is **synthetic demo data** so the app runs immediately. It is not genuine IPL match data. Do not claim its metrics represent real IPL prediction performance.
For a college presentation, replace it with verified historical IPL data, prepare innings-state rows correctly, and evaluate on later unseen matches.

## Setup (Windows / VS Code)
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal and optionally create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell activation is blocked, use Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Train the model:
   ```bash
   python train_model.py
   ```
6. Start the web app:
   ```bash
   streamlit run app.py
   ```
7. Open the local address printed in terminal (usually `http://localhost:8501`).

## Project structure
```text
IPL_Score_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── ipl_demo_data.csv
├── models/                 # created after training
└── src/
    ├── __init__.py
    └── predictor.py
```

## Model inputs
- current score
- overs completed
- wickets lost
- current run rate
- batting team
- bowling team
- venue category

Target: final innings score.

## Improvements for a proper college submission
1. Download a verified ball-by-ball IPL dataset.
2. Create one row for each innings state (for example, after each over).
3. Set target to the final score of that same innings.
4. Avoid data leakage: split by match ID or chronologically, not randomly across rows from the same match.
5. Compare with a simple baseline and report MAE/RMSE on unseen matches.
6. State the dataset source and season range in your report.

## Limitations
- Cricket scores are uncertain.
- Synthetic demo results do not prove real-world performance.
- The displayed +/- 15 range is illustrative, not a calibrated confidence interval.
- This is an educational tool, not a live-score service or betting advice.
