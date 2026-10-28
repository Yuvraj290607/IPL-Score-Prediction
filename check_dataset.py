
import pandas as pd
from pathlib import Path

# Project ke data folder ka path
DATA_PATH = Path(__file__).resolve().parent / "data"

# CSV files read karo
matches = pd.read_csv(DATA_PATH / "matches.csv")
deliveries = pd.read_csv(DATA_PATH / "deliveries.csv")

# Dataset ki basic information
print("\n========== MATCHES DATASET ==========")
print("Total matches:", len(matches))
print("Columns:", matches.columns.tolist())
print("\nFirst 5 matches:")
print(matches.head())

print("\n========== DELIVERIES DATASET ==========")
print("Total deliveries:", len(deliveries))
print("Columns:", deliveries.columns.tolist())
print("\nFirst 5 deliveries:")
print(deliveries.head())


print("\nMATCHES DATASET COLUMNS:")
print(matches.columns.tolist())

print("\nFIRST 5 MATCHES:")
print(matches.head().to_string())

