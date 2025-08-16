import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. LOAD THE PREDICTIONS CSV
#    This file should contain columns:
#    [Date, B365H, B365D, B365A, target,
#     RandomForest_Prediction, LogisticRegression_Prediction, XGBoost_Prediction,
#     SVM_Prediction, KNN_Prediction, NaiveBayes_Prediction, LightGBM_Prediction,
#     CatBoost_Prediction, Voting_Prediction]
# -------------------------------------------------------------------------
data = pd.read_csv("Data\predictions_test_data.csv", parse_dates=["Date"])

# Sort by Date just to be sure we're going chronologically
data.sort_values("Date", inplace=True)

# -------------------------------------------------------------------------
# 2. DEFINE MODEL COLUMNS (PREDICTIONS) & SET INITIAL BALANCES
# -------------------------------------------------------------------------
model_cols = [
    "RandomForest_Prediction",
    "LogisticRegression_Prediction",
    "XGBoost_Prediction",
    "SVC_Prediction",
    "KNN_Prediction",
    "NaiveBayes_Prediction",
    "LightGBM_Prediction",
    "CatBoost_Prediction",
    "Voting_Prediction",
]

initial_balance = 100.0  # Starting bankroll for each model
balances = {model: initial_balance for model in model_cols}

# Dictionary to store daily balance history for plotting
balance_history = {model: [] for model in model_cols}

# -------------------------------------------------------------------------
# 3. ITERATE DAY-BY-DAY, AND FOR EACH GAME, BET $1 IF THE MODEL HAS BALANCE
# -------------------------------------------------------------------------
unique_dates = sorted(data["Date"].unique())

for current_date in unique_dates:
    # Subset of games for this specific day
    day_games = data[data["Date"] == current_date]

    # For each model, bet on each game if the model has >= $1
    for model in model_cols:
        for _, game in day_games.iterrows():
            if balances[model] < 1:
                # If the model doesn't have at least $1, it can't bet anymore
                break

            # Model places a $1 bet
            balances[model] -= 1.0

            predicted_outcome = game[model]
            actual_outcome = game["target"]

            # If the prediction is correct, add winnings based on odds
            if predicted_outcome == actual_outcome:
                if predicted_outcome == 0:
                    odds = game["B365H"]
                elif predicted_outcome == 1:
                    odds = game["B365D"]
                elif predicted_outcome == 2:
                    odds = game["B365A"]
                else:
                    odds = 1.0  # fallback if unexpected

                # Add the winnings (1 * odds)
                balances[model] += odds

        # After finishing bets for today's matches, record the daily balance
        balance_history[model].append(balances[model])

# -------------------------------------------------------------------------
# 4. PRINT FINAL SUMMARY
# -------------------------------------------------------------------------
print("Final Balance Summary:")
for model in model_cols:
    print(f"{model}: ${balances[model]:.2f}")

# -------------------------------------------------------------------------
# 5. PLOT THE BALANCES OVER TIME
# -------------------------------------------------------------------------
# Convert the balance_history into a DataFrame for plotting
balance_df = pd.DataFrame(balance_history, index=unique_dates)
balance_df.index.name = "Date"

plt.figure(figsize=(12, 6))
for model in model_cols:
    plt.plot(balance_df.index, balance_df[model], marker="o", label=model)

plt.xlabel("Date")
plt.ylabel("Balance ($)")
plt.title("Model Balances Over Time")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
