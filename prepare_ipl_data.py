
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"

matches = pd.read_csv(DATA_PATH / "matches.csv")
deliveries = pd.read_csv(DATA_PATH / "deliveries.csv")

# Match results ko deliveries ke saath connect karo
matches = matches.rename(columns={"id": "match_id"})

# Sirf first innings ke deliveries lo
first_innings = deliveries[deliveries["inning"] == 1].copy()

# Har delivery ke baad running score aur wickets calculate karo
first_innings["current_score"] = (
    first_innings.groupby("match_id")["total_runs"].cumsum()
)

first_innings["wickets_lost"] = (
    first_innings.groupby("match_id")["is_wicket"].cumsum()
)

# Completed overs calculate karo
first_innings["overs_completed"] = (
    first_innings["over"] - 1
    + first_innings.groupby(["match_id", "over"]).cumcount().add(1) / 6
)

# Har match aur innings ka final score
final_scores = (
    first_innings.groupby("match_id")["total_runs"]
    .sum()
    .rename("final_score")
)

# Har match ke final score ko deliveries mein add karo
first_innings = first_innings.merge(
    final_scores,
    on="match_id",
    how="left"
)

# Matches se venue aur team information add karo
first_innings = first_innings.merge(
    matches[["match_id", "venue", "team1", "team2"]],
    on="match_id",
    how="left"
)

# Run rate calculate karo
first_innings["current_run_rate"] = (
    first_innings["current_score"]
    / first_innings["overs_completed"].replace(0, 0.1)
)

# Har 6 balls ke baad ek training example lo
first_innings["ball_number"] = (
    first_innings.groupby("match_id").cumcount() + 1
)

training_data = first_innings[
    (first_innings["ball_number"] % 6 == 0)
    & (first_innings["overs_completed"] < 20)
].copy()

# Required columns select karo
training_data = training_data[
    [
        "current_score",
        "overs_completed",
        "wickets_lost",
        "current_run_rate",
        "batting_team",
        "bowling_team",
        "venue",
        "final_score",
    ]
]

# Clean data
training_data = training_data.dropna()

# File save karo
output_path = DATA_PATH / "ipl_training_data.csv"
training_data.to_csv(output_path, index=False)

print("IPL training dataset created successfully!")
print("Total training rows:", len(training_data))
print("Columns:", training_data.columns.tolist())
print("Saved to:", output_path)
print(training_data.head())

print("\nFinal score statistics:")
print(first_innings["final_score"].describe())

print("\nUnique final scores:")
print(first_innings["final_score"].nunique())
