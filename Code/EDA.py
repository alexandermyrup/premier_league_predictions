import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/PL-games-14-24.csv')

# 1) Stack home and away performances into one table
home = df[['HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'team'})
home['win'] = home['FTR'] == 'H'

away = df[['AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'team'})
away['win'] = away['FTR'] == 'A'

all_matches = pd.concat([home, away], ignore_index=True)

# 2) Compute games played and wins per team
games_played = all_matches.groupby('team').size()
wins = all_matches.groupby('team')['win'].sum()

# 3) Compute win percentage
win_pct = (wins / games_played * 100).sort_values(ascending=False)

# 4) Plot
plt.figure(figsize=(10,6))
bars = plt.bar(win_pct.index, win_pct.values)
plt.xticks(rotation=90)
plt.ylabel('Win Percentage')
plt.title('Win Percentage by Team (with games played)')

# 5) Annotate each bar with vertical text centered in the bar
for bar, team in zip(bars, win_pct.index):
    count = games_played[team]
    x = bar.get_x() + bar.get_width() / 2       # horizontal center of bar
    y = bar.get_height() / 2                     # vertical middle of bar
    plt.text(
        x,
        y,
        f"{count} games",
        ha='center',                              # center horizontally
        va='center',                              # center vertically
        rotation=90,                              # vertical text
        fontsize=8,
        color='white'
    )

plt.tight_layout()
plt.show()