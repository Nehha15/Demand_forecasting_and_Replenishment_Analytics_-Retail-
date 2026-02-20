import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/processed/model_ready.csv", parse_dates=["Date"])

# -----------------------------
# Time-aware split
# -----------------------------
train = df[df["Date"] < "2012-01-01"].copy()
val = df[df["Date"] >= "2012-01-01"].copy()

y_val = val["Weekly_Sales"]


# -----------------------------
# Baseline predictions
# -----------------------------
val["naive_pred"] = val["lag_1"]
val["seasonal_naive_pred"] = val["lag_52"]
val["moving_avg_pred"] = val["rolling_mean_4"]


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# -----------------------------
# Overall Metrics
# -----------------------------
print("=== OVERALL PERFORMANCE ===")

for col in ["naive_pred", "seasonal_naive_pred", "moving_avg_pred"]:
    mae, rmse, mape = evaluate(y_val, val[col])
    print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")


# -----------------------------
# Holiday-only Evaluation
# -----------------------------
holiday_mask = val["IsHoliday"] == True

print("\n=== HOLIDAY WEEKS PERFORMANCE ===")

for col in ["naive_pred", "seasonal_naive_pred", "moving_avg_pred"]:
    mae, rmse, mape = evaluate(
        y_val[holiday_mask],
        val.loc[holiday_mask, col]
    )
    print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
