import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/processed/model_ready.csv", parse_dates=["Date"])

# -----------------------------
# Encode categorical features
# -----------------------------
df = pd.get_dummies(df, columns=["Type"], drop_first=True)


# -----------------------------
# Feature / Target split
# -----------------------------
TARGET = "Weekly_Sales"

FEATURES = [
    col for col in df.columns
    if col not in ["Weekly_Sales", "Date"]
]

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Time-aware split
# -----------------------------
train_idx = df["Date"] < "2012-01-01"

X_train, X_val = X[train_idx], X[~train_idx]
y_train, y_val = y[train_idx], y[~train_idx]

# -----------------------------
# Models
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# -----------------------------
# Train
# -----------------------------
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------
rf_pred = rf.predict(X_val)
gb_pred = gb.predict(X_val)


# -----------------------------
# Evaluate
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

print("RandomForest:", evaluate(y_val, rf_pred))
print("GradientBoosting:", evaluate(y_val, gb_pred))

# -----------------------------
# Feature Importance (Random Forest)
# -----------------------------
importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 15 Important Features:")
print(importances.head(15))


# ----- Holiday-only ML Evaluation

holiday_mask = X_val["IsHoliday"] == True

print("\nHoliday Performance:")

rf_h_mae = mean_absolute_error(
    y_val[holiday_mask],
    rf_pred[holiday_mask]
)

rf_h_rmse = np.sqrt(mean_squared_error(
    y_val[holiday_mask],
    rf_pred[holiday_mask]
))

print("RandomForest Holiday:", rf_h_mae, rf_h_rmse)
