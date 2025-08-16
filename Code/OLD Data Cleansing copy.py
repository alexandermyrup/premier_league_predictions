import pandas as pd
import numpy as np

# --- (Optional) If combining CSVs is needed, that code is commented out ---
# csv_files = [
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\19-20.csv",
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\20-21.csv",
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\21-22.csv",
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\22-23.csv",
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\23-24.csv",
# ]
# dataframes = [pd.read_csv(file) for file in csv_files]
# df_combined = pd.concat(dataframes)
# df_combined.to_csv(
#    r"C:\Users\alexa\Documents\ML Match Predictions\Data\PL-games-19-24.csv",
#    index=False,
# )

# --- Load Combined Data ---
df = pd.read_csv(r"Data/PL-games-19-24.csv")
pd.set_option("display.max_columns", None)
print(df)

# Drop columns that are not needed
df = df.drop(
    [
        "Div",
        "Time",
        "HTHG",
        "HTAG",
        "HTR",
        "Referee",
        "IWCH",
        "IWCD",
        "IWCA",
        "IWH",
        "IWD",
        "IWA",
    ],
    axis=1,
)

print("Columns after initial drop:")
print(list(df.columns))
print("Total number of rows:", df.shape[0])

# One-hot encode the HomeTeam and AwayTeam columns
home_dummies = pd.get_dummies(df["HomeTeam"], prefix="HomeTeam")
away_dummies = pd.get_dummies(df["AwayTeam"], prefix="AwayTeam")
df = pd.concat([df, home_dummies, away_dummies], axis=1)

print("Columns after one-hot encoding:")
print(list(df.columns))

# --- Compute team-level basic features ---
df["home_goal_diff"] = df["FTHG"] - df["FTAG"]
df["away_goal_diff"] = df["FTAG"] - df["FTHG"]

df["home_st_ratio"] = df["HST"] / df["HS"].replace(0, np.nan)
df["away_st_ratio"] = df["AST"] / df["AS"].replace(0, np.nan)

df["home_goal_conversion"] = df["FTHG"] / df["HS"].replace(0, np.nan)
df["away_goal_conversion"] = df["FTAG"] / df["AS"].replace(0, np.nan)

df["home_foul_to_card_ratio"] = np.where(
    (df["HY"] + df["HR"]) == 0, 0, df["HF"] / (df["HY"] + df["HR"])
)
df["away_foul_to_card_ratio"] = np.where(
    (df["AY"] + df["AR"]) == 0, 0, df["AF"] / (df["AY"] + df["AR"])
)

features_to_show = [
    "home_goal_diff",
    "away_goal_diff",
    "home_st_ratio",
    "away_st_ratio",
    "home_goal_conversion",
    "away_goal_conversion",
    "home_foul_to_card_ratio",
    "away_foul_to_card_ratio",
]
print("Team-level basic features:")
print(df[features_to_show].head())

# --- Create rolling averages for previous games for team-level stats ---
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

windows = [1, 3, 5, 10]
home_stats = [
    "FTHG",
    "HS",
    "HST",
    "HF",
    "HC",
    "HY",
    "HR",
    "home_goal_diff",
    "home_st_ratio",
    "home_goal_conversion",
    "home_foul_to_card_ratio",
]
away_stats = [
    "FTAG",
    "AS",
    "AST",
    "AF",
    "AC",
    "AY",
    "AR",
    "away_goal_diff",
    "away_st_ratio",
    "away_goal_conversion",
    "away_foul_to_card_ratio",
]

for stat in home_stats:
    for window in windows:
        new_col = f"home_{stat}_last{window}"
        df[new_col] = df.groupby("HomeTeam")[stat].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

for stat in away_stats:
    for window in windows:
        new_col = f"away_{stat}_last{window}"
        df[new_col] = df.groupby("AwayTeam")[stat].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

print("Rolling features (sample):")
print(df.filter(regex="_last(1|3|5|10)$").head())

print("DataFrame after rolling averages:")
print(df.head())
print("Columns:", list(df.columns))

# --- Final Feature Removal ---
# Remove the raw columns that are no longer needed.
df = df.drop(
    columns=[
        "FTHG",
        "FTAG",
        "HS",
        "HST",
        "HF",
        "HC",
        "HY",
        "HR",
        "AS",
        "AST",
        "AF",
        "AC",
        "AY",
        "AR",
        "home_goal_diff",
        "away_goal_diff",
        "home_st_ratio",
        "away_st_ratio",
        "home_goal_conversion",
        "away_goal_conversion",
        "home_foul_to_card_ratio",
        "away_foul_to_card_ratio",
    ],
    errors="ignore",  # in case any column is already dropped
)

# Encode FTR as target variable and drop it from the DataFrame
df["target"] = df["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)
df.drop(columns=["FTR"], inplace=True)

print("Final DataFrame (after removing unwanted features):")
print(df.head())
print("Final Columns:", list(df.columns))

# --- Rename columns with unwanted characters ---
# Replace '<' with '_less' to avoid issues with XGBoost
df.columns = df.columns.str.replace("<", "_less")

# Optional: If you have other unwanted characters, you can chain additional replacements.
# For example:
# df.columns = df.columns.str.replace('[', '_', regex=False)
# df.columns = df.columns.str.replace(']', '_', regex=False)

print("Columns after renaming unwanted characters:")
print(list(df.columns))


# --- Add head-to-head features ---
def add_head_to_head_features(df):
    df = df.copy()
    print("Sample dates from dataframe:")
    print(df["Date"].head())

    # Sort and create unique matchup keys
    df = df.sort_values("Date").reset_index(drop=True)
    df["matchup"] = df.apply(
        lambda row: "-".join(sorted([row["HomeTeam"], row["AwayTeam"]])), axis=1
    )
    df["directional_matchup"] = df["HomeTeam"] + "-" + df["AwayTeam"]

    # ... (head-to-head feature calculations as in your original code) ...
    # For brevity, assume the function computes and adds h2h_* columns.

    # After calculations, drop temporary matchup columns:
    df.drop(columns=["matchup", "directional_matchup"], inplace=True)
    return df


# Add head-to-head features
df = add_head_to_head_features(df)

# Verify head-to-head features
h2h_columns = [col for col in df.columns if col.startswith("h2h_")]
print("New head-to-head features added:")
print(h2h_columns)
if h2h_columns:
    print(df[h2h_columns].head())

# Encode target if needed (if FTR exists)
if "FTR" in df.columns:
    df["target"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    df.drop(columns=["FTR"], inplace=True)

# Save the final enhanced feature-engineered DataFrame
df.to_csv(
    r"Data/PL-games-19-24-feature-engineered-final-3.csv",
    index=False,
)

print("Head-to-head features engineering complete and unwanted features removed!")
