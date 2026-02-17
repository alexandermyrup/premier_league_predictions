import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
# Suppress LightGBM info and warning logs
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- Load data ---
df = pd.read_csv("Data/Processed/PL-games-19-24-feature-engineered-final-3-normalised.csv", parse_dates=["Date"]) \
    .dropna().sort_values("Date")
odds = pd.read_csv("Data/Processed/PL-games-19-24-feature-engineered-final-3.csv", parse_dates=["Date"]) \
    .dropna().sort_values("Date")

# --- Train/Validation split (encapsulated) ---
# If you want to use a separate upcoming-games file as the test set, set upcoming_test_path to a CSV path.
upcoming_test_path = None  # e.g. "Data/upcoming_games.csv"

def create_splits(df, odds, upcoming_test_path=None, train_frac=0.7, valid_frac=0.85):
    """Return X_train, y_train, X_valid, y_valid, odds_valid, X_test, y_test, odds_test

    If upcoming_test_path is provided, that file is used as X_test/odds_test (it may not contain 'target').
    Otherwise X_test/odds_test will be None.
    """
    n = len(df)
    t = int(n * train_frac)
    v = int(n * valid_frac)

    X_train = df.iloc[:t].drop(["Date","HomeTeam","AwayTeam","target"], axis=1)
    y_train = df.iloc[:t]["target"].values
    X_valid = df.iloc[t:v].drop(["Date","HomeTeam","AwayTeam","target"], axis=1)
    y_valid = df.iloc[t:v]["target"].values
    odds_valid = odds.iloc[t:v][["B365H","B365D","B365A"]].values

    X_test = None
    y_test = None
    odds_test = None
    if upcoming_test_path:
        up = pd.read_csv(upcoming_test_path, parse_dates=["Date"])  # expect same column names
        # keep only rows with features present
        odds_test = up[["B365H","B365D","B365A"]].values if all(c in up.columns for c in ["B365H","B365D","B365A"]) else None
        drop_cols = [c for c in ["Date","HomeTeam","AwayTeam","target"] if c in up.columns]
        X_test = up.drop(columns=drop_cols, errors='ignore')
        if 'target' in up.columns:
            y_test = up['target'].values

    return X_train, y_train, X_valid, y_valid, odds_valid, X_test, y_test, odds_test


# create splits now
X_train, y_train, X_valid, y_valid, odds_valid, X_test, y_test, odds_test = create_splits(df, odds, upcoming_test_path)
print(f"Shapes -> X_train: {X_train.shape}, y_train: {getattr(y_train,'shape', None)}, X_valid: {X_valid.shape}, y_valid: {getattr(y_valid,'shape', None)}")
if X_test is not None:
    print(f"Loaded upcoming test set: X_test {X_test.shape}, y_test {getattr(y_test,'shape', None)}, odds_test: {None if odds_test is None else odds_test.shape}")

# --- simulate ROI for Flat betting (stake=1 each game) ---
def simulate_roi_flat(preds, odds, actuals):
    total_return = 0.0
    total_bets = 0
    for pred, odd, actual in zip(preds, odds, actuals):
        total_bets += 1
        if pred == actual:
            # add only the winning odd for the predicted outcome
            total_return += odd[pred]
    # ROI: (total returned - total stakes) / total stakes
    return (total_return - total_bets) / total_bets if total_bets > 0 else -1

# --- objective: maximize Flat ROI via model choice and hyperparameters ---
def objective(trial):
    model_name = trial.suggest_categorical('model', ['RF','XGB','LGBM','Cat','LR','SVM','KNN','NB','Voting'])
    if model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('rf_n', 50,300),
            'max_depth': trial.suggest_int('rf_d', 3,20),
            'random_state': 42
        }
        clf = RandomForestClassifier(**params)
    elif model_name == 'XGB':
        params = {
            'n_estimators': trial.suggest_int('xgb_n', 50,300),
            'max_depth': trial.suggest_int('xgb_d', 3,15),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01,0.3),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        clf = XGBClassifier(**params)
    elif model_name == 'LGBM':
        params = {
            'n_estimators': trial.suggest_int('lgb_n', 50,300),
            'max_depth': trial.suggest_int('lgb_d', 3,15),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01,0.3),
            'random_state': 42
        }
        clf = lgb.LGBMClassifier(**params)
    elif model_name == 'Cat':
        params = {
            'iterations': trial.suggest_int('cb_n', 50,300),
            'depth': trial.suggest_int('cb_d', 3,12),
            'learning_rate': trial.suggest_float('cb_lr', 0.01,0.3),
            'random_state': 42,
            'verbose': False
        }
        clf = CatBoostClassifier(**params)
    elif model_name == 'LR':
        C = trial.suggest_float('lr_C', 1e-4,10, log=True)
        clf = LogisticRegression(C=C, solver='liblinear', max_iter=500)
    elif model_name == 'SVM':
        C = trial.suggest_float('svm_C', 1e-3,100, log=True)
        gamma = trial.suggest_float('svm_g', 1e-4,1, log=True)
        clf = SVC(C=C, gamma=gamma, probability=True)
    elif model_name == 'KNN':
        k = trial.suggest_int('knn_k', 3,30)
        clf = KNeighborsClassifier(n_neighbors=k)
    elif model_name == 'NB':
        clf = GaussianNB()
    else:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(C=1, solver='liblinear')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ]
        clf = VotingClassifier(estimators=estimators, voting='soft')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)
    roi = simulate_roi_flat(preds, odds_valid, y_valid)
    # ensure scalar return
    roi = float(roi)
    return roi

# --- run optimization with tqdm progress bar and ETA ---
n_trials = 900
study = optuna.create_study(direction='maximize')
pbar = tqdm(total=n_trials, desc="Optuna Trials")

def progress_callback(study, trial):
    pbar.update()

study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
pbar.close()
print('Overall best trial:', study.best_trial.params)

# --- Summarize best parameters per model ---
best_per_model = {}
for t in study.get_trials(deepcopy=False):
    if t.value is None:
        continue
    m = t.params['model']
    if m not in best_per_model or t.value > best_per_model[m]['value']:
        best_per_model[m] = {'value': t.value, 'params': t.params}

# Convert best_per_model to a DataFrame for easy viewing
import pandas as pd
rows = []
for model, info in best_per_model.items():
    params = {k: v for k, v in info['params'].items() if k != 'model'}
    row = {'model': model, 'best_roi': info['value'], **params}
    rows.append(row)
df_best = pd.DataFrame(rows).sort_values('model')

print('Best parameters per model:')
print(df_best.to_string(index=False))

# Optionally save to CSV
output_csv = 'Data/Output/best_parameters_per_model.csv'
df_best.to_csv(output_csv, index=False)
print(f'Saved best parameters per model to {output_csv}')

# --- Summarize best parameters per model ---
best_per_model = {}
for t in study.get_trials(deepcopy=False):
    if t.value is None:
        continue
    m = t.params['model']
    # Initialize or update if this trial is better
    if m not in best_per_model or t.value > best_per_model[m]['value']:
        best_per_model[m] = {'value': t.value, 'params': t.params}

print('Best ROI per model:')
for m, info in best_per_model.items():
    print(f"Model: {m}")
    print(f"  Best ROI: {info['value']:.4f}")
    # Filter out the 'model' key from params when printing
    params = {k: v for k, v in info['params'].items() if k != 'model'}
    print(f"  Best params: {params}")
