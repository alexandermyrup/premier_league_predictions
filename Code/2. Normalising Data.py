import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the DataFrame
df = pd.read_csv("Data/Processed/PL-games-19-24-feature-engineered-final-3.csv")


# Define the target column
target_col = "target"

# Identify numeric feature columns for scaling (excluding the target column)
feature_cols = (
    df.drop(columns=[target_col]).select_dtypes(include=["float64", "int64"]).columns
)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler and transform only the feature columns
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print(df.head())

# Save the normalized DataFrame to a CSV file
df.to_csv("Data/Processed/PL-games-19-24-feature-engineered-final-3-normalised.csv", index=False)

print("Normalized data saved successfully!")
