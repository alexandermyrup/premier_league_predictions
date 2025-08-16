import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize
import joblib

# for training models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ---------------------------------------------------
# 1) Read Data
# ---------------------------------------------------
DATA_PATH = Path("Data/PL-games-19-24-feature-engineered-final-3-normalised.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at {DATA_PATH.resolve()}")

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data.sort_values("Date", inplace=True)

# ---------------------------------------------------
# 2) Ensure probability columns are numeric
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
    if data[col].apply(lambda x: isinstance(x, list)).any():
        data[col] = data[col].apply(lambda x: x[0] if isinstance(x, list) and x else np.nan)
    data[col] = data[col].astype(str).str.strip('[]')
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
# 5) Simulation parameters & stats tracking
# ---------------------------------------------------
STRATEGIES = ["Flat", "Threshold", "Value", "Kelly"]
INITIAL_BALANCE = 100.0
THRESHOLD_VALUE = 0.6
FRACTIONAL_KELLY = 0.3

balance_dict = {s: {m: INITIAL_BALANCE for m in model_cols} for s in STRATEGIES}
history_dict = {s: {m: [] for m in model_cols} for s in STRATEGIES}
stats_dict = {
    s: {
        m: {"bets": 0, "wins": 0, "stakes": 0.0}
        for m in model_cols
    }
    for s in STRATEGIES
}

dates_sorted = sorted(data["Date"].unique())

# ---------------------------------------------------
# X) Annualization factor from your one-year test set
# ---------------------------------------------------
periods_per_year = len(dates_sorted)
print(f"Annualizing Sharpe over {periods_per_year} bet-days per year")

# ---------------------------------------------------
# 6) Main simulation loop (fills history_dict & stats_dict)
# ---------------------------------------------------
for current_date in dates_sorted:
    games = data[data["Date"] == current_date]
    for strategy in STRATEGIES:
        for model in model_cols:
            bal_start = balance_dict[strategy][model]
            if bal_start < 1e-9 or (strategy in ("Threshold","Value","Kelly") and model not in prob_based_models):
                history_dict[strategy][model].append(bal_start)
                continue

            bets = []
            for _, game in games.iterrows():
                pred = game[model]
                actual = game["target"]
                odds = offered_odds(game, pred)
                prob = game.get(model.replace("_Prediction", "_Probability"), np.nan)
                stake = desired_stake(
                    strategy, bal_start, odds, prob,
                    threshold=THRESHOLD_VALUE,
                    fractionK=FRACTIONAL_KELLY
                )
                if stake <= 0 or pd.isna(odds):
                    continue
                bets.append({"stake": stake, "win": pred == actual, "odds": odds})

            total_stake = sum(b["stake"] for b in bets)
            if total_stake > bal_start and total_stake > 0:
                scale = bal_start / total_stake
                for b in bets:
                    b["stake"] *= scale
                total_stake = bal_start

            bal_end = bal_start - total_stake
            for b in bets:
                stats = stats_dict[strategy][model]
                stats["bets"] += 1
                stats["stakes"] += b["stake"]
                if b["win"]:
                    stats["wins"] += 1
                    bal_end += b["stake"] * b["odds"]

            balance_dict[strategy][model] = bal_end
            history_dict[strategy][model].append(bal_end)

# ---------------------------------------------------
# 7) Plot & save equity curves
# ---------------------------------------------------
graphs_dir = Path("Graphs")
graphs_dir.mkdir(parents=True, exist_ok=True)

for strat in STRATEGIES:
    hist_df = pd.DataFrame(history_dict[strat], index=dates_sorted)
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in model_cols:
        ax.plot(hist_df.index, hist_df[model], marker="o", linewidth=1, label=model)
    ax.set_title(f"{strat} Strategy – Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fn = graphs_dir / f"{strat.lower()}_equity_curve.png"
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Equity curves saved in: {graphs_dir.resolve()}")

# ---------------------------------------------------
# 8) Summary metrics & CSV export (with annualized Sharpe)
# ---------------------------------------------------
metrics = []
for strat in STRATEGIES:
    for model in model_cols:
        stats = stats_dict[strat][model]
        bets   = stats["bets"]
        wins   = stats["wins"]
        stakes = stats["stakes"]
        win_pct = (wins / bets * 100) if bets > 0 else np.nan
        net_profit = balance_dict[strat][model] - INITIAL_BALANCE
        roi = net_profit / INITIAL_BALANCE

        eq = pd.Series(history_dict[strat][model], index=dates_sorted)
        rets = eq.pct_change().dropna()
        if not rets.empty:
            raw_sharpe = rets.mean() / rets.std()
            sharpe = raw_sharpe * np.sqrt(periods_per_year)
        else:
            sharpe = np.nan

        running_max = eq.cummax()
        drawdowns = (running_max - eq) / running_max
        max_dd = drawdowns.max()

        metrics.append({
            "Strategy":       strat,
            "Model":          model,
            "Bets Placed":    bets,
            "Win %":          win_pct,
            "Total Staked":   stakes,
            "Net Profit":     net_profit,
            "ROI":            roi,
            "Sharpe":         sharpe,
            "Max Drawdown":   max_dd
        })

metrics_df = pd.DataFrame(metrics)
out_csv = graphs_dir / "strategy_model_summary.csv"
metrics_df.to_csv(out_csv, index=False)
print(f"Summary metrics written to: {out_csv.resolve()}")

# ---------------------------------------------------
# 9) ROI grouped bar chart
# ---------------------------------------------------
bar_data = metrics_df.pivot(index="Model", columns="Strategy", values="ROI")

fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.2
x = np.arange(len(bar_data.index))

for i, strat in enumerate(bar_data.columns):
    ax.bar(x + i * bar_width,
           bar_data[strat] * 100,
           width=bar_width,
           label=strat)

ax.set_xticks(x + bar_width * (len(bar_data.columns) - 1) / 2)
ax.set_xticklabels(bar_data.index, rotation=45, ha="right")
ax.set_ylabel("ROI (%)")
ax.set_title("ROI by Model and Strategy (on initial 100 units)")
ax.legend(title="Strategy")

plt.tight_layout()
bar_fn = graphs_dir / "roi_grouped_bar.png"
fig.savefig(bar_fn, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"ROI grouped bar chart saved to: {bar_fn.resolve()}")

# ---------------------------------------------------
# 10) ROI heatmap with smooth blending
# ---------------------------------------------------
pivot = metrics_df.pivot(index="Strategy", columns="Model", values="ROI")

vmin = pivot.min().min()
vmax = pivot.max().max()
range_ = vmax - vmin if vmax != vmin else 1.0

def frac(x):
    return (x - vmin) / range_

p_neg  = frac(-0.05)
p_zero = frac(0.0)
p_pos  = frac(0.05)
p_turq = frac(0.20)

stops = [
    (0.0,    "red"),
    (p_neg,  "red"),
    (p_zero, "orange"),
    (p_pos,  "green"),
    (p_turq, "turquoise"),
    (1.0,    "turquoise"),
]
cmap = LinearSegmentedColormap.from_list("roi_cmap", stops)
norm = Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(14, 6))
cax = ax.imshow(pivot, aspect="auto", cmap=cmap, norm=norm)

for i, strat in enumerate(pivot.index):
    for j, model in enumerate(pivot.columns):
        val = pivot.iloc[i, j]
        txt = f"{val:.1%}" if not np.isnan(val) else ""
        ax.text(j, i, txt, ha="center", va="center")

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_title("ROI Heatmap: Strategy vs. Model")
fig.colorbar(cax, ax=ax, label="ROI", fraction=0.046, pad=0.04)

plt.tight_layout()
heatmap_fn = graphs_dir / "roi_heatmap.png"
fig.savefig(heatmap_fn, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"ROI heatmap saved to: {heatmap_fn.resolve()}")

# ---------------------------------------------------
# 11) Train & save models for feature importances
# ---------------------------------------------------
TRAIN_PATH = Path("Data/training_data.csv")
if TRAIN_PATH.exists():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    # assume raw features plus 'target' and 'Date'
    X_train = train.drop(columns=["target", "Date"])
    y_train = train["target"]

    # ensure Models/ directory exists
    models_dir = Path("Models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # fit & save each
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
    joblib.dump(rf, models_dir / "random_forest.pkl")

    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    joblib.dump(logreg, models_dir / "logistic_regression.pkl")

    xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)
    joblib.dump(xg, models_dir / "xgboost.pkl")

    lgbm = lgb.LGBMClassifier().fit(X_train, y_train)
    joblib.dump(lgbm, models_dir / "lightgbm.pkl")

    cb = CatBoostClassifier(verbose=0).fit(X_train, y_train)
    joblib.dump(cb, models_dir / "catboost.pkl")

    print("Models trained and saved to Models/")

else:
    print(f"Training data not found at {TRAIN_PATH.resolve()}, skipping training step.")

# ---------------------------------------------------
# 12) Feature Importances & CSV export (top 20 features per model)
# ---------------------------------------------------
model_paths = {
    "RandomForest":      "Models/random_forest.pkl",
    "LogisticRegression":"Models/logistic_regression.pkl",
    "XGBoost":           "Models/xgboost.pkl",
    "LightGBM":          "Models/lightgbm.pkl",
    "CatBoost":          "Models/catboost.pkl"
}

feature_importances_list = []

for model_name, model_path in model_paths.items():
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}, skipping {model_name}.")
        continue

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = model.feature_names_in_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
        feature_names = model.feature_names_in_
    else:
        print(f"No feature importances available for {model_name}, skipping.")
        continue

    fi_df = pd.DataFrame({
        "Model":      model_name,
        "Feature":    feature_names,
        "Importance": importances
    })
    top20 = fi_df.sort_values("Importance", ascending=False).head(20)
    feature_importances_list.append(top20)

if feature_importances_list:
    feat_imp_df = pd.concat(feature_importances_list, ignore_index=True)
    feat_imp_csv = graphs_dir / "feature_importances_top20.csv"
    feat_imp_df.to_csv(feat_imp_csv, index=False)
    print(f"Top 20 feature importances saved to: {feat_imp_csv.resolve()}")
else:
    print("No feature importances were extracted.")
