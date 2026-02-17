import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------
# 1) Read Data
# ---------------------------------------------------
DATA_PATH = Path("Data/Output/predictions_test_data_normalised.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at {DATA_PATH.resolve()}")

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data.sort_values("Date", inplace=True)

# ---------------------------------------------------
# 2) Ensure probability columns are numeric
#    Handle list or string representations and convert to float
# ---------------------------------------------------
float_cols = [
    "RandomForest_Probability",
    "LogisticRegression_Probability",
    "XGBoost_Probability",
    "SVM_Probability",
    "KNN_Probability",
    "NaiveBayes_Probability",
    "LightGBM_Probability",
    "CatBoost_Probability",
    "Voting_Probability"
]

for col in float_cols:
    if col not in data.columns:
        continue
    # unwrap lists
    if data[col].apply(lambda x: isinstance(x, list)).any():
        data[col] = data[col].apply(lambda x: x[0] if isinstance(x, list) and x else np.nan)
    # strip brackets if still string
    data[col] = data[col].astype(str).str.strip('[]')
    # convert to float
    data[col] = pd.to_numeric(data[col], errors='raise')

# ---------------------------------------------------
# 3) Model & column definitions
# ---------------------------------------------------
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
prob_based_models = [
    m for m in model_cols if m.replace("_Prediction", "_Probability") in float_cols
]

# ---------------------------------------------------
# 4) Helper functions
# ---------------------------------------------------
def kelly_fraction(prob: float, odds: float) -> float:
    if pd.isna(prob) or pd.isna(odds) or prob <= 0 or odds <= 1:
        return 0.0
    b = odds - 1.0
    f_raw = (prob * b - (1 - prob)) / b
    return max(0.0, min(f_raw, 1.0))

def desired_stake(strategy: str, balance: float, odds: float, prob: float,
                  *, threshold: float = 0.6, fractionK: float = 1.0) -> float:
    if balance <= 0:
        return 0.0
    if strategy == "Flat":
        return 1.0
    if strategy == "Threshold":
        return 1.0 if prob >= threshold else 0.0
    if strategy == "Value":
        fair = 1.0 / prob if prob > 0 else np.inf
        return 1.0 if odds > fair else 0.0
    if strategy == "Kelly":
        f = kelly_fraction(prob, odds) * fractionK
        return balance * f
    raise ValueError(f"Unknown strategy {strategy}")

ODDS_COLS = {0: "B365H", 1: "B365D", 2: "B365A"}

def offered_odds(game_row: pd.Series, pred: int) -> float:
    return game_row.get(ODDS_COLS.get(pred), np.nan)

# ---------------------------------------------------
# 5) Simulation parameters
# ---------------------------------------------------
STRATEGIES = ["Flat", "Threshold", "Value", "Kelly"]
INITIAL_BALANCE = 100.0
THRESHOLD_VALUE = 0.6
FRACTIONAL_KELLY = 0.3

balance_dict = {s: {m: INITIAL_BALANCE for m in model_cols} for s in STRATEGIES}
history_dict = {s: {m: [] for m in model_cols} for s in STRATEGIES}
dates_sorted = sorted(data["Date"].unique())

# ---------------------------------------------------
# 6) Main simulation loop
# ---------------------------------------------------
for current_date in dates_sorted:
    games = data[data["Date"] == current_date]
    for strategy in STRATEGIES:
        for model in model_cols:
            bal_start = balance_dict[strategy][model]
            if bal_start < 1e-9:
                history_dict[strategy][model].append(bal_start)
                continue
            if strategy in ("Threshold", "Value", "Kelly") and model not in prob_based_models:
                history_dict[strategy][model].append(bal_start)
                continue
            bets = []
            for _, game in games.iterrows():
                pred = game[model]
                actual = game["target"]
                odds = offered_odds(game, pred)
                prob = game.get(model.replace("_Prediction", "_Probability"), np.nan)
                stake = desired_stake(strategy, bal_start, odds, prob,
                                       threshold=THRESHOLD_VALUE,
                                       fractionK=FRACTIONAL_KELLY)
                if stake <= 0 or pd.isna(odds):
                    continue
                bets.append({"stake": stake, "win": pred == actual, "odds": odds})
            total_stake = sum(b["stake"] for b in bets)
            # scale stakes to bankroll if exceeding balance
            if total_stake > bal_start and total_stake > 0:
                scale = bal_start / total_stake
                for b in bets:
                    b["stake"] *= scale
                total_stake = bal_start
            bal_end = bal_start - total_stake
            for b in bets:
                if b["win"]:
                    bal_end += b["stake"] * b["odds"]
            balance_dict[strategy][model] = bal_end
            history_dict[strategy][model].append(bal_end)

# ---------------------------------------------------
# 7) Summary & plots
# ---------------------------------------------------
for strat in STRATEGIES:
    print(f"\n=== {strat} Betting Final Balances ===")
    for model in model_cols:
        print(f"{model}: ${balance_dict[strat][model]:.2f}")
fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(11, 16), sharex=True)
for i, strat in enumerate(STRATEGIES):
    ax = axes[i]
    ax.set_title(f"{strat} Strategy – Bankroll Trajectories")
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance ($)")
    hist_df = pd.DataFrame(history_dict[strat], index=dates_sorted)
    for model in model_cols:
        ax.plot(hist_df.index, hist_df[model], marker="o", linewidth=1, label=model)
    ymin, ymax = hist_df.min().min(), hist_df.max().max()
    ax.set_ylim(max(0, ymin * 0.95), ymax * 1.05 if ymax > 0 else 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
