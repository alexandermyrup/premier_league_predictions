# exploring different models with Platt calibration
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV  # for calibration
from sklearn.model_selection import TimeSeriesSplit

# ----- Data Loading and Preprocessing -----
df = pd.read_csv("Data/PL-games-19-24-feature-engineered-final-3-normalised.csv", parse_dates=["Date"])
df2 = pd.read_csv("Data/PL-games-19-24-feature-engineered-final-3.csv", parse_dates=["Date"])

# Dropna & Alignment: ensure same rows dropped in both dataframes to keep features & outputs aligned
mask = df.dropna().index.intersection(df2.dropna().index)
df = df.loc[mask].sort_values("Date").reset_index(drop=True)
df2 = df2.loc[mask].sort_values("Date").reset_index(drop=True)

# 80/20 chronological split
split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]
x2_test = df2.iloc[split:]  # aligned

# Prepare output dataframe
output_df = x2_test[["Date", "B365H", "B365D", "B365A", "target"]].copy()

# Features and target
y_train = train_df["target"]
y_test  = test_df["target"]
X_train = train_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
X_test  = test_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])

target_names = ["home win", "draw", "away win"]

# TimeSeriesSplit for calibration CV
tscv = TimeSeriesSplit(n_splits=5)

# Model training & Platt calibration function
def train_and_calibrate(base_clf, X_train, y_train, cv):
    clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=cv)
    clf.fit(X_train, y_train)
    return clf

# Instantiate base models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=188, max_depth=19, min_samples_split=7, random_state=42),
    "LogisticRegression": LogisticRegression(C=0.0107, solver="liblinear", max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=224, max_depth=7, learning_rate=0.0715, use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "SVM": SVC(C=43.7064, gamma=0.01156, kernel="rbf", probability=True, random_state=42),
    # KNN: skip calibration to avoid segfaults
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="manhattan"),
    "NaiveBayes": GaussianNB(),
    "LightGBM": lgb.LGBMClassifier(n_estimators=286, max_depth=4, num_leaves=53, learning_rate=0.23099, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=432, depth=4, learning_rate=0.20843, l2_leaf_reg=6.8016, random_state=42, verbose=0)
}

# Calibrate/train models
calibrated = {}
for name, base in models.items():
    if name == "SVM":
        # SVM already includes Platt scaling via probability=True
        calibrated[name] = base.fit(X_train, y_train)
    elif name == "KNN":
        # Skip calibration for KNN to avoid segmentation faults
        calibrated[name] = base.fit(X_train, y_train)
    else:
        calibrated[name] = train_and_calibrate(base, X_train, y_train, tscv)

# Predict and evaluate
for name, clf in calibrated.items():
    preds = clf.predict(X_test)
    print(f"--- {name} Classification Report ---")
    print(classification_report(y_test, preds, target_names=target_names))
