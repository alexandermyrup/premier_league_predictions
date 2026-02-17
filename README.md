# Premier League Predictions

Machine learning pipeline for predicting English Premier League match outcomes and evaluating betting strategy profitability. Built as part of a Bachelor Thesis at Copenhagen Business School.

**Research Question:** *How effective are machine learning models in generating sustainable profits within the football betting market?*

**Authors:** Alexander Myrup & Sebastian Ehrhardt
**Supervisor:** Professor Daniel Hardt
**Date:** May 2025

The full paper is included in `Paper.pdf`.

## Project Overview

This project goes beyond prediction accuracy to answer whether ML models can generate positive ROI through sports betting. It trains 9 classifiers on 5 seasons of Premier League data (2019-2024), then simulates 4 betting strategies across all models to measure profitability.

### Pipeline

```
1. Data Cleansing    → Combine season CSVs, drop irrelevant columns, one-hot encode teams
2. Normalising Data  → StandardScaler on all numeric features
2.5 ROI Tuning      → Optuna hyperparameter search (900 trials, optimising for ROI)
3. Model Training    → Train 8 models + voting ensemble, generate predictions
4. Betting Strategies→ Simulate Flat, Threshold, Value, and Kelly betting
5. Reporting Models  → Confusion matrices, ROC curves, metrics table
6. Reporting Betting → Equity curves, ROI heatmap, Sharpe ratios, drawdowns
```

### Models

| Model | Accuracy | Best ROI Strategy |
|-------|----------|-------------------|
| Random Forest | 54.8% | Value Betting |
| CatBoost | 51.9% | Kelly Criterion |
| Logistic Regression | 51.4% | Kelly Criterion |
| Voting Ensemble | 51.2% | Kelly Criterion |
| KNN | 50.8% | Flat Betting |
| LightGBM | 49.4% | Flat Betting |
| SVM | 49.2% | Flat Betting |
| XGBoost | 50.1% | Flat Betting |
| Naive Bayes | 45.9% | Flat Betting |
| Dummy (always Home) | 39.6% | — |

### Betting Strategies

- **Flat**: Fixed $1 stake on every prediction
- **Threshold**: Only bet when model confidence >= 60%
- **Value**: Only bet when bookmaker odds exceed fair odds (1/predicted probability)
- **Kelly Criterion**: Stake proportional to perceived edge (fractional Kelly = 0.3)

## Repo Structure

```
Code/
  1. Data Cleansing.py          # Combine CSVs, feature engineering, rolling averages
  2. Normalising Data.py        # StandardScaler normalisation
  2.5 ROI Tuning.py             # Optuna hyperparameter optimisation
  3. Model Training.py          # Train models, generate predictions & metrics
  4. Betting Strategies.py      # Simulate 4 betting strategies
  5. Reporting Models.py        # Confusion matrices, ROC curves
  6. Reporting Betting Strategies.py  # Equity curves, ROI heatmap, Sharpe/drawdown
  EDA.py                        # Exploratory data analysis
  Feature importance.py         # Feature importance extraction

Data/
  Raw/                          # Season CSVs (19-20 through 24-25)
  Processed/                    # Pipeline intermediates (combined, feature-engineered, normalised)
  Output/                       # Predictions, metrics, strategy summaries

Graphs/                         # Confusion matrices, ROC curves, equity curves, ROI visualisations
Paper.pdf                       # Full Bachelor Thesis
```

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — 5 seasons, ~1,900 matches.

**Feature engineering** produces ~130 features:
- One-hot encoded team identities (home & away)
- Rolling averages (last 1, 3, 5, 10 games) for goals, shots, fouls, corners, cards
- Derived ratios: shot-on-target ratio, goal conversion rate, foul-to-card ratio
- Bookmaker odds from Bet365, Pinnacle, and market aggregates

## Requirements

- Python 3.13+
- pandas, numpy, matplotlib, scikit-learn
- xgboost, lightgbm, catboost
- optuna (for hyperparameter tuning)

## Running the Pipeline

Scripts are numbered and should be run in order from the repo root:

```bash
python "Code/1. Data Cleansing.py"
python "Code/2. Normalising Data.py"
python "Code/3. Model Training.py"
python "Code/4. Betting Strategies.py"
python "Code/5. Reporting Models.py"
python "Code/6. Reporting Betting Strategies.py"
```

## Key Findings

Prediction accuracy alone does not determine betting profitability. The Voting ensemble achieved the strongest combined ROI across strategies, while models like XGBoost showed that lower accuracy can still yield competitive returns when paired with the right betting strategy. The Kelly Criterion amplified both gains and losses, making it high-risk/high-reward. Value Betting and Flat Betting provided more stable returns.

## Next Steps

- **Fix data leakage in normalisation**: StandardScaler is currently fitted on the full dataset before the train/test split, meaning test data statistics leak into training. The scaler should be fitted on training data only, then applied to the test set separately.
- **Apply to live data**: Extend the pipeline to predict upcoming fixtures using current-season data and real-time odds, enabling actual betting decisions rather than purely historical backtesting.
- **Expand the dataset**: Test whether including more historical seasons (10+ years) improves or hurts model performance, as discussed in the paper.
- **Improve calibration**: Explore probability calibration techniques (Platt scaling, isotonic regression) to improve the reliability of predicted probabilities, which directly affects Value and Kelly betting strategies.
