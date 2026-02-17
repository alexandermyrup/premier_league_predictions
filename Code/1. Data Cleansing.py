import pandas as pd
import numpy as np
import re

# --- (Optional) If combining CSVs is needed, that code is commented out ---
def combine_csvs():
    csv_files = [
        r"Data/Raw/19-20.csv",
        r"Data/Raw/20-21.csv",
        r"Data/Raw/21-22.csv",
        r"Data/Raw/22-23.csv",
        r"Data/Raw/23-24.csv",
        r"Data/Raw/24-25.csv",
    ]
    dataframes = [pd.read_csv(file) for file in csv_files]
    df_combined = pd.concat(dataframes)
    df_combined.to_csv(
        r"Data/Processed/PL-games-19-24.csv",
        index=False,
    )

#combine_csvs()

# --- Load Combined Data and normalize column names (fix NBSP, zero-width, comma decimals, extra spaces)
df = pd.read_csv("Data/Processed/PL-games-19-24.csv")
pd.set_option("display.max_columns", None)

orig_cols = list(df.columns)
normalized = []
for c in orig_cols:
    nc = c.replace("\u00A0", " ")      # NBSP -> space
    nc = nc.replace("\u200b", "")      # zero-width space -> nothing
    nc = nc.replace(",", ".")           # comma decimal -> dot
    nc = re.sub(r"\s+", " ", nc)       # collapse multiple spaces
    nc = nc.strip()
    normalized.append(nc)
df.columns = normalized
# normalize '<' to '_less' early so detection picks up variants like 'B365C<2.5'
df.columns = [c.replace('<', '_less') for c in df.columns]

# Build drop list and detect columns by exact normalized match or by suspicious substrings
orig_drop = ["Div","Time","HTHG","HTAG","HTR","Referee", "BWH", 'BWD', "BWA",
             "IWCH","IWCD","IWCA","IWH","IWD","IWA", "BFH", "BFD", "BFA", "1XBH"]
drop_norm = [c.replace("\u00A0", " ").replace(",", ".").strip() for c in orig_drop]

# Substrings to catch: WH/VC/BW patterns and 1.00/BFE variants requested by user
suspicious_substrs = [
    "WHCH","WHCD","WHCA","VCCH","VCCD","VCCA",
    "BWCH","BWCD","BWCA","WHH","WHD","WHA",
    "VCH","VCD","VCA",
    "1.00 XBD","1.00 XBA","BFE_less2.5","BFEC_less2.5",
    "BFE","BFEC",
    # additional columns requested to drop
    "B365CH","B365CD","B365CA","PSCH","PSCD","PSCA",
    "MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA",
    "B365C>2.5","B365C_less2.5","PC>2.5","PC_less2.5",
    "MaxC>2.5","MaxC_less2.5","AvgC>2.5","AvgC_less2.5",
    "AHCh","B365CAHH","B365CAHA","PCAHH","PCAHA","MaxCAHH","MaxCAHA","AvgCAHH","AvgCAHA",
    "BFCH","BFCD","BFCA","1XBCH","1XBCD","1XBCA",
]

exact_matches = [c for c in df.columns if c in drop_norm]
contains_matches = [c for c in df.columns if any(sub in c for sub in suspicious_substrs)]
to_drop = sorted(set(exact_matches + contains_matches))

print("Column drop diagnostics:")
print("  total columns:", len(df.columns))
print("  exact normalized drop matches:", exact_matches)
print("  contains-pattern matches:", contains_matches)
if to_drop:
    print("Dropping columns:", to_drop)
    df.drop(columns=to_drop, errors='ignore', inplace=True)
else:
    print("No columns matched for dropping (check normalization/drop list)")

df.drop(columns=to_drop, errors='ignore', inplace=True)

# One-hot encode teams
df = pd.concat([df, pd.get_dummies(df["HomeTeam"], prefix="HomeTeam")], axis=1)
df = pd.concat([df, pd.get_dummies(df["AwayTeam"], prefix="AwayTeam")], axis=1)

# Team-level basic features
df["home_goal_diff"] = df["FTHG"] - df["FTAG"]
df["away_goal_diff"] = df["FTAG"] - df["FTHG"]
df["home_st_ratio"] = df["HST"] / df["HS"].replace(0, np.nan)
df["away_st_ratio"] = df["AST"] / df["AS"].replace(0, np.nan)
df["home_goal_conversion"] = df["FTHG"] / df["HS"].replace(0, np.nan)
df["away_goal_conversion"] = df["FTAG"] / df["AS"].replace(0, np.nan)
df["home_foul_to_card_ratio"] = np.where((df["HY"]+df["HR"])==0, 0, df["HF"]/(df["HY"]+df["HR"]))
df["away_foul_to_card_ratio"] = np.where((df["AY"]+df["AR"])==0, 0, df["AF"]/(df["AY"]+df["AR"]))

# Date sorting
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.sort_values("Date", inplace=True)

# Rolling averages for team stats
windows = [1,3,5,10]
home_stats = ["FTHG","HS","HST","HF","HC","HY","HR",
              "home_goal_diff","home_st_ratio","home_goal_conversion","home_foul_to_card_ratio"]
away_stats = ["FTAG","AS","AST","AF","AC","AY","AR",
              "away_goal_diff","away_st_ratio","away_goal_conversion","away_foul_to_card_ratio"]
for stat_list, team_col, prefix in [(home_stats, "HomeTeam", "home"), (away_stats, "AwayTeam", "away")]:
    for stat in stat_list:
        for w in windows:
            col_name = f"{prefix}_{stat}_last{w}"
            df[col_name] = df.groupby(df[team_col])[stat] \
                .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())

# --- Add head-to-head features (before dropping raw columns) --- Not used anymore
def add_head_to_head_features(df, windows=[1,3,5]):
    df = df.copy()
    df.sort_values('Date', inplace=True)
    df['directional_matchup'] = df['HomeTeam'] + '-' + df['AwayTeam']
    for w in windows:
        df[f'h2h_home_goals_last{w}'] = df.groupby('directional_matchup')['FTHG'] \
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        df[f'h2h_away_goals_last{w}'] = df.groupby('directional_matchup')['FTAG'] \
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        df[f'h2h_home_wins_last{w}'] = df.groupby('directional_matchup')['target'] \
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1) \
                       .apply(lambda y: (y==0).sum(), raw=False))
        df[f'h2h_away_wins_last{w}'] = df.groupby('directional_matchup')['target'] \
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1) \
                       .apply(lambda y: (y==2).sum(), raw=False))
        df[f'h2h_draws_last{w}'] = df.groupby('directional_matchup')['target'] \
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1) \
                       .apply(lambda y: (y==1).sum(), raw=False))
    df.drop(columns=['directional_matchup'], inplace=True)
    return df

# Encode target and remove raw match outcomes after H2H features
# Must create target before dropping FTR
if 'FTR' in df.columns:
    df['target'] = df['FTR'].map({'H':0,'D':1,'A':2}).astype(int)

# Apply head-to-head features
#df = add_head_to_head_features(df)

# Now drop raw score and count columns
raw_cols = ['FTHG','FTAG','HS','HST','HF','HC','HY','HR','AS','AST','AF','AC','AY','AR',
            'home_goal_diff','away_goal_diff','home_st_ratio','away_st_ratio',
            'home_goal_conversion','away_goal_conversion','home_foul_to_card_ratio','away_foul_to_card_ratio','FTR']
df.drop(columns=raw_cols, errors='ignore', inplace=True)

# Clean column names
df.columns = df.columns.str.replace("<", "_less")

# Save final feature-engineered data
df.to_csv(r"Data/Processed/PL-games-19-24-feature-engineered-final-3.csv", index=False)

print("Head-to-head features added and DataFrame saved.")
