import streamlit as st
import pandas as pd
from predictor import load_model, predict_score, load_training_data


st.set_page_config(page_title="IPL Score Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Score Prediction System")
st.caption("A learning project using Python, Pandas, Scikit-learn and Streamlit.")
st.info("This demo estimates a first-innings total from the current match situation. It is an educational model, not a live betting tool or a guarantee of the final score.")

model, feature_columns = load_model()
data = load_training_data()
if model is None:
    st.error("The model has not been trained yet. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Enter match situation")
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    ]
    batting_team = st.selectbox("Batting team", teams)
    bowling_team = st.selectbox("Bowling team", [team for team in teams if team != batting_team])
    venue = st.selectbox(
    "Venue",
    sorted(data["venue"].dropna().unique()) if data is not None and "venue" in data.columns
    else ["M Chinnaswamy Stadium"]
)
    current_score = st.number_input("Current score (runs)", min_value=0, max_value=350, value=92, step=1)
    overs = st.slider("Completed overs", min_value=1, max_value=19, value=10)
    balls_in_over = st.slider("Balls completed in current over", min_value=0, max_value=5, value=0)
    wickets = st.slider("Wickets lost", min_value=0, max_value=9, value=2)
    total_balls = overs * 6 + balls_in_over
    if total_balls >= 120:
        st.warning("A standard T20 first innings has 120 legal balls. Choose fewer overs.")
        st.stop()
    overs_completed = total_balls / 6
    remaining_balls = 120 - total_balls
    current_run_rate = current_score / overs_completed if overs_completed > 0 else 0
    remaining_overs = remaining_balls / 6
    st.write(f"**Current run rate:** {current_run_rate:.2f}")
    st.write(f"**Overs remaining:** {remaining_overs:.1f}")
    predict_button = st.button("Predict final score", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if predict_button:
        result = predict_score(
            model, feature_columns, current_score, overs_completed, wickets,
            current_run_rate, batting_team, bowling_team, venue
        )
        predicted = result["predicted_score"]
        low, high = result["range"]
        st.metric("Estimated final score", f"{predicted} runs")
        st.write(f"Illustrative planning range: **{low}–{high} runs**")
        st.caption("The range is a simple uncertainty band, not a statistically guaranteed interval.")
        st.progress(min(max(int(predicted / 250 * 100), 0), 100), text="Score estimate on a 0–250 display scale")
        st.markdown("#### How the estimate is formed")
        st.write("The trained regression model uses current score, overs completed, wickets lost, run rate, teams and venue category.")
    else:
        st.write("Enter the match situation on the left, then click **Predict final score**.")
        st.metric("Estimated final score", "—")

st.divider()
st.subheader("Training data preview")
data = load_training_data()
if data is not None and not data.empty:
    st.write(f"Rows available for training: **{len(data)}**")
    st.dataframe(data.head(10), use_container_width=True)
    st.caption("The included CSV is synthetic demo data created to make the app runnable immediately. Replace it with verified historical IPL data before reporting real-world performance.")
else:
    st.warning("Training CSV not found.")

st.subheader("Project workflow")
st.markdown("1. Load innings-state data.  2. Train a regression model.  3. Enter match details.  4. Generate an estimate.  5. Evaluate using MAE and RMSE.")

st.subheader("Project workflow")
st.markdown("1. Load innings-state data.  2. Train a regression model.  3. Enter match details.  4. Generate an estimate.  5. Evaluate using MAE and RMSE.")


# -------------------------------
# IPL DATA VISUALIZATION
# -------------------------------

st.divider()
st.header("📊 IPL Data Analysis Dashboard")



if data is not None and not data.empty:

    # Chart 1: Final score distribution
    st.subheader("1. Final Score Distribution")

    score_chart = data["final_score"].dropna()

    st.bar_chart(
        score_chart.value_counts().sort_index(),
        x_label="Final Score",
        y_label="Number of innings"
    )

    # Chart 2: Average score by batting team
    st.subheader("2. Average Score by Batting Team")

    team_scores = (
        data.groupby("batting_team")["final_score"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        team_scores,
        x_label="Batting Team",
        y_label="Average Final Score"
    )

    # Chart 3: Overs completed vs final score
    st.subheader("3. Overs Completed vs Final Score")

    st.scatter_chart(
        data,
        x="overs_completed",
        y="final_score",
        x_label="Overs Completed",
        y_label="Final Score"
    )

    # Chart 4: Wickets lost vs final score
    st.subheader("4. Wickets Lost vs Final Score")

    wicket_scores = (
        data.groupby("wickets_lost")["final_score"]
        .mean()
    )

    st.line_chart(
        wicket_scores,
        x_label="Wickets Lost",
        y_label="Average Final Score"
    )

else:
    st.warning("Training dataset is not available.")

