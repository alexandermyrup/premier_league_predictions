import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from catboost import CatBoostClassifier

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
        solver="saga",
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
        probability=True,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=14,
        weights="uniform",
        metric="manhattan",
        n_jobs=-1
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

# Train each base model
dict_models = {}
for name, clf in models.items():
    dict_models[name] = clf.fit(X_train_arr, y_train)

# Add the always-0 dummy classifier
dummy_clf = DummyClassifier(strategy="constant", constant=0)
dummy_clf.fit(X_train_arr, y_train)
dict_models["Dummy"] = dummy_clf

# Build & train soft‐voting ensemble (excluding the dummy)
voting_clf = VotingClassifier(
    estimators=[(n, m) for n, m in dict_models.items() if n != "Dummy"],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_arr, y_train)
dict_models["Voting"] = voting_clf

# ─── Collect predictions & probabilities ───
prob_store = {}
pred_store = {}
for name, clf in dict_models.items():
    probs = clf.predict_proba(X_test_arr)
    preds = np.argmax(probs, axis=1)
    prob_store[name] = probs
    pred_store[name] = preds

    output_df[f"{name}_Prediction"] = preds
    output_df[f"{name}_Probability"] = np.max(probs, axis=1)

# Save predictions CSV
output_df.to_csv("Data/predictions_test_data_normalised.csv", index=False)

# ─── Compute metrics ───
y_test_bin = label_binarize(y_test, classes=target_classes)
metrics = []
for name in dict_models:
    preds = pred_store[name]
    probs = prob_store[name]
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

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Generate and save confusion matrices for each model
for name, preds in pred_store.items():
    cm = confusion_matrix(y_test, preds, labels=target_classes)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.colorbar()
    tick_marks = np.arange(len(target_classes))
    plt.xticks(tick_marks, target_classes)
    plt.yticks(tick_marks, target_classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f"graphs/confusion_{name}.png")
    plt.close()

# Plot macro-average ROC curve for all models
plt.figure()
n_classes = len(target_classes)

for name, probs in prob_store.items():
    # 1) Compute per-class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 2) Aggregate all FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 3) Interpolate each TPR at those FPRs and average
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    # 4) Compute macro AUC on the averaged curve
    macro_auc = auc(all_fpr, mean_tpr)

    # 5) Plot it
    plt.plot(all_fpr, mean_tpr,
             label=f'{name} (macro AUC = {macro_auc:.2f})')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve for All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("graphs/roc_all_models_macro.png")
plt.show()
