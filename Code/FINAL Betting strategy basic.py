import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data and parse the Date column
data = pd.read_csv("Data\predictions_test_data_normalised.csv", parse_dates=["Date"])
data.sort_values("Date", inplace=True)

# List of prediction model columns
model_cols = [
    "RandomForest_Prediction",
    "LogisticRegression_Prediction",
    "XGBoost_Prediction",
    "SVM_Prediction",
    "KNN_Prediction",
    "NaiveBayes_Prediction",
    "LightGBM_Prediction",
    "CatBoost_Prediction",
    "Voting_Prediction",
]

# Initialize each model's balance and a history dictionary for plotting
initial_balance = 100
balances = {model: initial_balance for model in model_cols}
balance_history = {model: [] for model in model_cols}

# Get a sorted list of unique dates
unique_dates = sorted(data["Date"].unique())

# Iterate over each day in the dataset
for current_date in unique_dates:
    # Filter games for the current day in the order they appear
    day_games = data[data["Date"] == current_date]

    # Process each model independently if it has sufficient balance
    for model in model_cols:
        # Only bet on games if balance is at least $1
        for idx, game in day_games.iterrows():
            if balances[model] < 1:
                # Stop betting for this model if balance is insufficient
                break

            # Place a bet: subtract $1
            balances[model] -= 1

            # Get the model's prediction and the actual target outcome
            prediction = game[model]
            target = game["target"]

            # If the prediction is correct, add the winnings based on the corresponding odds
            if prediction == target:
                if prediction == 0:
                    odds = game["B365H"]
                elif prediction == 1:
                    odds = game["B365D"]
                elif prediction == 2:
                    odds = game["B365A"]
                else:
                    odds = 1  # Fallback in case of an unexpected value

                balances[model] += 1 * odds  # Add winnings

        # After processing the day's games, record the balance.
        # If the model is out (balance < 1), it remains at that value for subsequent days.
        balance_history[model].append(balances[model])

# Create a final summary of balances
final_summary = {model: balances[model] for model in model_cols}
print("Final Balance Summary:")
for model, balance in final_summary.items():
    print(f"{model}: ${balance:.2f}")

# Convert the balance history into a DataFrame for plotting
balance_df = pd.DataFrame(balance_history, index=unique_dates)
balance_df.index.name = "Date"

# Plot the balances over time
plt.figure(figsize=(12, 6))
for model in model_cols:
    plt.plot(balance_df.index, balance_df[model], marker="o", label=model)
plt.xlabel("Date")
plt.ylabel("Balance ($)")
plt.title("Model Balances Over Time")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
