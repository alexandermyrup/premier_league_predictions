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
df = pd.read_csv(r"Data\PL-games-19-24.csv")
print(list(df.columns))
