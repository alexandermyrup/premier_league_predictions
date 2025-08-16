# exploring different models
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

# ----- Data Loading and Preprocessing -----
df = pd.read_csv(
    r"Data\PL-games-19-24-feature-engineered-final-3.csv",
    parse_dates=["Date"],
)

print("Original data shape:", df.shape)

df = df.dropna()

print("Shape after dropna:", df.shape)

# Sort chronologically
df = df.sort_values("Date")

# 80/20 chronological split
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
print(train_df.shape, test_df.shape)

# Create a new DataFrame with the desired test columns
output_df = test_df[["Date", "B365H", "B365D", "B365A", "target"]].copy()

# Define features and target
X_train = train_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
X_test = test_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "target"])
y_train = train_df["target"]
y_test = test_df["target"]

# Define target names explicitly since the target is already encoded
target_names = ["home win", "draw", "away win"]

#############################
# Random Forest Classifier  #
#############################
print("----- Random Forest Classifier -----")

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\n")

#################################
# Logistic Regression Classifier#
#################################
print("----- Logistic Regression -----")

lr_clf = LogisticRegression(random_state=42, max_iter=1000)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\n")

# Define the unwanted characters
unwanted_chars = ["[", "]", "<"]

# Find columns in X_train that contain any unwanted character
columns_with_unwanted = [
    col for col in X_train.columns if any(char in col for char in unwanted_chars)
]

print("Columns with unwanted characters:", columns_with_unwanted)

#############################
# XGBoost Classifier        #
#############################
print("----- XGBoost -----")

xgb_clf = XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\n")

#############################
# Support Vector Machine    #
#############################
print("----- Support Vector Machine (SVM) -----")

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\n")

#############################
# K-Nearest Neighbors (KNN) #
#############################
print("----- K-Nearest Neighbors (KNN) -----")

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)
print("Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\n")

#############################
# Naive Bayes               #
#############################
print("----- Naive Bayes -----")

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\n")

#############################
# LightGBM Classifier       #
#############################
print("----- LightGBM -----")


lgb_clf = lgb.LGBMClassifier(random_state=42)
lgb_clf.fit(X_train, y_train)
y_pred_lgb = lgb_clf.predict(X_test)

accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("LightGBM Accuracy:", accuracy_lgb)
print("Classification Report:")
print(classification_report(y_test, y_pred_lgb, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lgb))
print("\n")


#############################
# CatBoost Classifier       #
#############################
print("----- CatBoost -----")

catboost_clf = CatBoostClassifier(random_state=42, verbose=0)
catboost_clf.fit(X_train, y_train)
y_pred_catboost = catboost_clf.predict(X_test)

accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print("CatBoost Accuracy:", accuracy_catboost)
print("Classification Report:")
print(classification_report(y_test, y_pred_catboost, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_catboost))
print("\n")

#############################
# Summary of Base Accuracies#
#############################
print("----- Summary of Base Accuracies -----")
results = {
    "Random Forest": accuracy_rf,
    "Logistic Regression": accuracy_lr,
    "XGBoost": accuracy_xgb,
    "SVM": accuracy_svm,
    "KNN": accuracy_knn,
    "Naive Bayes": accuracy_nb,
    "LightGBM": accuracy_lgb,
    "CatBoost": accuracy_catboost,
    #    "FNN": fnn_acc,
    #    "DNN": dnn_acc
}

for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

#############################
# Append Predictions & Voting Model #
#############################
# Add prediction columns for each model
output_df["RandomForest_Prediction"] = y_pred_rf
output_df["LogisticRegression_Prediction"] = y_pred_lr
output_df["XGBoost_Prediction"] = y_pred_xgb
output_df["SVM_Prediction"] = y_pred_svm
output_df["KNN_Prediction"] = y_pred_knn
output_df["NaiveBayes_Prediction"] = y_pred_nb
output_df["LightGBM_Prediction"] = y_pred_lgb
output_df["CatBoost_Prediction"] = y_pred_catboost

# Use the csv created predictions to compute the voting prediction
prediction_columns = [
    "RandomForest_Prediction",
    "LogisticRegression_Prediction",
    "XGBoost_Prediction",
    "SVM_Prediction",
    "KNN_Prediction",
    "NaiveBayes_Prediction",
    "LightGBM_Prediction",
    "CatBoost_Prediction",
]

# For each row, count the number of 0's, 1's, and 2's and choose the most common
output_df["Voting_Prediction"] = output_df[prediction_columns].apply(
    lambda row: row.value_counts().idxmax(), axis=1
)

# Output the DataFrame to a CSV file
output_df.to_csv("Data\predictions_test_data.csv", index=False)
