import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier


# ----- Data Loading and Preprocessing -----
df = pd.read_csv(
    "Data/PL-games-19-24-feature-engineered-final-3-normalised.csv",
    parse_dates=["Date"]
)
df2 = pd.read_csv(
    "Data/PL-games-19-24-feature-engineered-final-3.csv",
    parse_dates=["Date"]
)

drop_idx = df.dropna().index.intersection(df2.dropna().index)
df = df.loc[drop_idx].sort_values("Date").reset_index(drop=True)
df2 = df2.loc[drop_idx].sort_values("Date").reset_index(drop=True)

split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]
x2_test = df2.iloc[split:]

# Prepare output DataFrame from raw odds and true target
output_df = x2_test[["Date", "B365H", "B365D", "B365A", "target"]].copy()

y_train = train_df["target"]
y_test = test_df["target"]
X_train = train_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
X_test = test_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])

# convert to numpy once
X_train_arr = X_train.values
X_test_arr = X_test.values

target_classes = [0, 1, 2]

# Instantiate base models with parallelism
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=142,
        max_depth=16,
        min_samples_split=7,
        n_jobs=-1,
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=0.007745580273217738,
        solver="saga",        # supports n_jobs
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=53,
        max_depth=8,
        learning_rate=0.24220207101841673,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    ),
    "SVM": SVC(
        C=2.5327930787079693,
        gamma=0.013113313028176306,
        kernel="rbf",
        probability=True,     # still expensive, but kept for soft voting
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=14,
        weights="uniform",
        metric="manhattan",
        n_jobs=-1             # fit isn't parallel, but predict_proba is
    ),
    "NaiveBayes": GaussianNB(),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.2575179289238998,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=183,
        depth=12,
        learning_rate=0.13655235092720033,
        l2_leaf_reg=6.8016,
        random_state=42,
        verbose=0
    )
}

# Train each model
dict_models = {}
for name, clf in models.items():
    dict_models[name] = clf.fit(X_train_arr, y_train)


# Instantiate & fit the always-0 dummy
dummy_clf = DummyClassifier(strategy="constant", constant=0)
dummy_clf.fit(X_train_arr, y_train)
dict_models["Dummy"] = dummy_clf

# Build your VotingClassifier, excluding "Dummy"
voting_clf = VotingClassifier(
    estimators=[(n, m) for n, m in dict_models.items() if n != "Dummy"],
    voting="soft",
    n_jobs=-1
)
voting_clf.fit(X_train_arr, y_train)
dict_models["Voting"] = voting_clf

# ─── Collect predictions & probabilities ───
prob_store = {}
for name, clf in dict_models.items():
    probs = clf.predict_proba(X_test_arr)
    prob_store[name] = probs
    preds = np.argmax(probs, axis=1)
    prob_of_pred = np.max(probs, axis=1)

    output_df[f"{name}_Prediction"] = preds
    output_df[f"{name}_Probability"] = prob_of_pred

# Save predictions CSV
output_df.to_csv("Data/predictions_test_data_normalised.csv", index=False)

# ─── Compute metrics ───
y_test_bin = label_binarize(y_test, classes=target_classes)

metrics = []
for name, clf in dict_models.items():
    preds = output_df[f"{name}_Prediction"]
    probs = prob_store[name]   # reuse instead of recomputing

    metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, average='macro'),
        'Recall': recall_score(y_test, preds, average='macro'),
        'F1': f1_score(y_test, preds, average='macro'),
        'AUC': roc_auc_score(y_test_bin, probs, multi_class='ovr', average='macro'),
        'Brier': np.mean(np.sum((probs - y_test_bin) ** 2, axis=1))
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("Data/model_metrics.csv", index=False)
print(metrics_df)
