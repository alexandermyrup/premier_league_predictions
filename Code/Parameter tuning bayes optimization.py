import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from catboost import CatBoostClassifier

# ===== 1. LOAD AND PREPARE DATA =====
df = pd.read_csv(
    r"Data\PL-games-19-24-feature-engineered-final-3.csv",
    parse_dates=["Date"],
)
print("Original data shape:", df.shape)

df = df.dropna()
print("Shape after dropna:", df.shape)

df = df.sort_values("Date")

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
print("Train/Test split:", train_df.shape, test_df.shape)

# For final output
output_df = test_df[["Date", "B365H", "B365D", "B365A", "target"]].copy()

X_train = train_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
y_test = test_df["target"]

target_names = ["home win", "draw", "away win"]


# ===== 2. DEFINE HELPER FUNCTION FOR OPTUNA TUNING =====
def tune_model(objective_func, n_trials=30):
    """
    objective_func: a function(trial) -> float that Optuna will optimize
    n_trials: how many trials Optuna will run
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=n_trials)
    print("  Best Score:", study.best_value)
    print("  Best Params:", study.best_params)
    return study.best_params


# We’ll define separate objective functions for each model,
# each returning a cross-validation (CV) score.

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ===== 3. OBJECTIVE FUNCTIONS FOR EACH MODEL =====


def objective_rf(trial):
    # Suggest hyperparameters for RandomForest
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_lr(trial):
    # Suggest hyperparams for LogisticRegression
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
    # Make sure solver can handle l1 if chosen
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    # If penalty='l1' and solver='saga' or 'liblinear', it's valid
    # But 'l1' is not compatible with solver='lbfgs' or 'newton-cg', etc.

    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42
    )

    # Some combos might be invalid (e.g. penalty='l1', solver='liblinear' is valid,
    # but penalty='l1', solver='newton-cg' is not). We handle that:
    try:
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()
    except ValueError:
        # Invalid combination of penalty/solver
        return 0.0


def objective_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 2, 12)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_svc(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    # gamma relevant for rbf, poly, sigmoid
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_knn(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int(
        "p", 1, 2
    )  # Minkowski with p=1 (Manhattan) or p=2 (Euclidean)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_nb(trial):
    # GaussianNB has mainly var_smoothing
    var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-5, log=True)

    model = GaussianNB(var_smoothing=var_smoothing)

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_lgb(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 64, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)

    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


def objective_cat(trial):
    depth = trial.suggest_int("depth", 2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    iterations = trial.suggest_int("iterations", 50, 500, step=50)

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        verbose=0,
        random_state=42,
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()


# ===== 4. RUN OPTIMIZATION FOR EACH MODEL =====
print("\n=== Tuning Random Forest ===")
best_params_rf = tune_model(objective_rf, n_trials=30)

print("\n=== Tuning Logistic Regression ===")
best_params_lr = tune_model(objective_lr, n_trials=30)

print("\n=== Tuning XGBoost ===")
best_params_xgb = tune_model(objective_xgb, n_trials=30)

print("\n=== Tuning SVC ===")
best_params_svc = tune_model(objective_svc, n_trials=30)

print("\n=== Tuning KNN ===")
best_params_knn = tune_model(objective_knn, n_trials=30)

print("\n=== Tuning GaussianNB ===")
best_params_nb = tune_model(objective_nb, n_trials=30)

print("\n=== Tuning LightGBM ===")
best_params_lgb = tune_model(objective_lgb, n_trials=30)

print("\n=== Tuning CatBoost ===")
best_params_cat = tune_model(objective_cat, n_trials=30)

# ===== 5. TRAIN FINAL MODELS ON BEST PARAMS AND EVALUATE =====
# Re-initialize each model with best_params and train on the entire X_train

print("\n===== Final Model Training & Evaluation =====")

# Random Forest
rf_clf = RandomForestClassifier(**best_params_rf, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\n--- Random Forest ---")
print("Test Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf, target_names=target_names))
print(confusion_matrix(y_test, y_pred_rf))

# Logistic Regression
lr_clf = LogisticRegression(**best_params_lr, random_state=42)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("\n--- Logistic Regression ---")
print("Test Accuracy:", accuracy_lr)
print(classification_report(y_test, y_pred_lr, target_names=target_names))
print(confusion_matrix(y_test, y_pred_lr))

# XGBoost
xgb_clf = XGBClassifier(
    **best_params_xgb, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("\n--- XGBoost ---")
print("Test Accuracy:", accuracy_xgb)
print(classification_report(y_test, y_pred_xgb, target_names=target_names))
print(confusion_matrix(y_test, y_pred_xgb))

# SVC
svc_clf = SVC(**best_params_svc, random_state=42)
svc_clf.fit(X_train, y_train)
y_pred_svc = svc_clf.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("\n--- SVC ---")
print("Test Accuracy:", accuracy_svc)
print(classification_report(y_test, y_pred_svc, target_names=target_names))
print(confusion_matrix(y_test, y_pred_svc))

# KNN
knn_clf = KNeighborsClassifier(**best_params_knn)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\n--- KNN ---")
print("Test Accuracy:", accuracy_knn)
print(classification_report(y_test, y_pred_knn, target_names=target_names))
print(confusion_matrix(y_test, y_pred_knn))

# GaussianNB
nb_clf = GaussianNB(**best_params_nb)
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\n--- Naive Bayes ---")
print("Test Accuracy:", accuracy_nb)
print(classification_report(y_test, y_pred_nb, target_names=target_names))
print(confusion_matrix(y_test, y_pred_nb))

# LightGBM
lgb_clf = lgb.LGBMClassifier(**best_params_lgb, random_state=42)
lgb_clf.fit(X_train, y_train)
y_pred_lgb = lgb_clf.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("\n--- LightGBM ---")
print("Test Accuracy:", accuracy_lgb)
print(classification_report(y_test, y_pred_lgb, target_names=target_names))
print(confusion_matrix(y_test, y_pred_lgb))

# CatBoost
cat_clf = CatBoostClassifier(**best_params_cat, random_state=42, verbose=0)
cat_clf.fit(X_train, y_train)
y_pred_cat = cat_clf.predict(X_test)
accuracy_cat = accuracy_score(y_test, y_pred_cat)
print("\n--- CatBoost ---")
print("Test Accuracy:", accuracy_cat)
print(classification_report(y_test, y_pred_cat, target_names=target_names))
print(confusion_matrix(y_test, y_pred_cat))

# ===== 6. SUMMARY OF ALL ACCURACIES =====
print("\n===== Summary of Tuned Accuracies =====")
results = {
    "Random Forest": accuracy_rf,
    "Logistic Regression": accuracy_lr,
    "XGBoost": accuracy_xgb,
    "SVC": accuracy_svc,
    "KNN": accuracy_knn,
    "Naive Bayes": accuracy_nb,
    "LightGBM": accuracy_lgb,
    "CatBoost": accuracy_cat,
}
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# ===== 7. ENSEMBLE (MAJORITY VOTING) =====
output_df["RandomForest_Prediction"] = y_pred_rf
output_df["LogisticRegression_Prediction"] = y_pred_lr
output_df["XGBoost_Prediction"] = y_pred_xgb
output_df["SVC_Prediction"] = y_pred_svc
output_df["KNN_Prediction"] = y_pred_knn
output_df["NaiveBayes_Prediction"] = y_pred_nb
output_df["LightGBM_Prediction"] = y_pred_lgb
output_df["CatBoost_Prediction"] = y_pred_cat

prediction_columns = [
    "RandomForest_Prediction",
    "LogisticRegression_Prediction",
    "XGBoost_Prediction",
    "SVC_Prediction",
    "KNN_Prediction",
    "NaiveBayes_Prediction",
    "LightGBM_Prediction",
    "CatBoost_Prediction",
]

# Majority vote: pick the most frequent prediction among the models
output_df["Voting_Prediction"] = output_df[prediction_columns].apply(
    lambda row: row.value_counts().idxmax(), axis=1
)

output_df.to_csv("Data/predictions_test_data.csv", index=False)
print("\nSaved predictions to Data/predictions_test_data.csv")
