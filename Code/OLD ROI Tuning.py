import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load and preprocess data
df = (
    pd.read_csv(
        "Data/PL-games-19-24-feature-engineered-final-3-normalised.csv",
        parse_dates=["Date"],
    )
    .dropna()
    .sort_values("Date")
)
odds_df = (
    pd.read_csv(
        "Data/PL-games-19-24-feature-engineered-final-3.csv", parse_dates=["Date"]
    )
    .dropna()
    .sort_values("Date")
)

train_idx = int(len(df) * 0.7)
valid_idx = int(len(df) * 0.85)

X_train = df.iloc[:train_idx].drop(["Date", "HomeTeam", "AwayTeam", "target"], axis=1)
y_train = df.iloc[:train_idx]["target"]

X_valid = df.iloc[train_idx:valid_idx].drop(
    ["Date", "HomeTeam", "AwayTeam", "target"], axis=1
)
y_valid = df.iloc[train_idx:valid_idx]["target"].values
odds_valid = odds_df.iloc[train_idx:valid_idx][["B365H", "B365D", "B365A"]].values


# ROI Calculation Function
def calculate_roi(preds, odds, actuals, stake=1):
    total_staked, total_return = 0, 0
    for pred, odd, actual in zip(preds, odds, actuals):
        chosen_odd = odd[pred]
        total_staked += stake
        if pred == actual:
            total_return += stake * chosen_odd
    return (total_return - total_staked) / total_staked if total_staked else -1


# Optimization Functions for each model
def optimize_model(model_name, model_cls, param_space, trials=50):
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return calculate_roi(preds, odds_valid, y_valid)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, catch=(Exception,))
    print(f"\n{model_name} best params: {study.best_params}")


# Random Forest
optimize_model(
    "Random Forest",
    RandomForestClassifier,
    {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 500),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "random_state": lambda t: 42,
    },
)

# XGBoost
optimize_model(
    "XGBoost",
    XGBClassifier,
    {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 500),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
        "random_state": lambda t: 42,
        "use_label_encoder": lambda t: False,
        "eval_metric": lambda t: "mlogloss",
    },
)

# LightGBM
optimize_model(
    "LightGBM",
    lgb.LGBMClassifier,
    {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 500),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 10, 150),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
        "random_state": lambda t: 42,
        "verbose": lambda t: -1,
    },
)

# CatBoost
optimize_model(
    "CatBoost",
    CatBoostClassifier,
    {
        "iterations": lambda t: t.suggest_int("iterations", 50, 500),
        "depth": lambda t: t.suggest_int("depth", 3, 12),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1, 10),
        "random_state": lambda t: 42,
        "verbose": lambda t: False,
    },
)

# SVM
optimize_model(
    "SVM",
    SVC,
    {
        "C": lambda t: t.suggest_loguniform("C", 1e-3, 1e2),
        "gamma": lambda t: t.suggest_loguniform("gamma", 1e-4, 1e1),
        "kernel": lambda t: t.suggest_categorical("kernel", ["rbf", "poly"]),
        "random_state": lambda t: 42,
    },
)

# Logistic Regression
optimize_model(
    "Logistic Regression",
    LogisticRegression,
    {
        "C": lambda t: t.suggest_loguniform("C", 1e-4, 1e2),
        "solver": lambda t: t.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        "max_iter": lambda t: 1000,
        "random_state": lambda t: 42,
    },
)

# KNN
optimize_model(
    "KNN",
    KNeighborsClassifier,
    {
        "n_neighbors": lambda t: t.suggest_int("n_neighbors", 3, 30),
        "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": lambda t: t.suggest_categorical(
            "metric", ["euclidean", "manhattan", "minkowski"]
        ),
    },
)

# Naive Bayes (no hyperparameters)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds_nb = nb_model.predict(X_valid)
roi_nb = calculate_roi(preds_nb, odds_valid, y_valid)
print(f"\nNaive Bayes ROI: {roi_nb:.4f}")
