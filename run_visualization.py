import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.replenishment import simulate_inventory
import os

# Ensure figure folder exists
os.makedirs("reports/figures", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/model_ready.csv", parse_dates=["Date"])
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

TARGET = "Weekly_Sales"
FEATURES = [col for col in df.columns if col not in ["Weekly_Sales", "Date"]]

train_idx = df["Date"] < "2012-01-01"
train = df[train_idx]
val = df[~train_idx].copy()

X_train = train[FEATURES]
y_train = train[TARGET]
X_val = val[FEATURES]

# Train RF
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

val["rf_pred"] = rf.predict(X_val)
val["naive_pred"] = val["lag_1"]

# Select one Store & Dept
example = val[(val["Store"] == 1) & (val["Dept"] == 1)]

plt.figure(figsize=(12,6))
plt.plot(example["Date"], example["Weekly_Sales"], label="Actual")
plt.plot(example["Date"], example["rf_pred"], label="RandomForest")
plt.plot(example["Date"], example["naive_pred"], label="Naive")
plt.legend()
plt.title("Actual vs Forecast (Store 1, Dept 1)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("reports/figures/actual_vs_forecast.png")
plt.close()

print("Saved actual_vs_forecast.png")

importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(15)

plt.figure(figsize=(8,6))
importances.sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances (RandomForest)")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png")
plt.close()

print("Saved feature_importance.png")



rf_results = simulate_inventory(val, "rf_pred")
naive_results = simulate_inventory(val, "naive_pred")

rf_mean = rf_results.mean()
naive_mean = naive_results.mean()

comparison = pd.DataFrame({
    "RandomForest": [rf_mean["HoldingCost"], rf_mean["StockoutCost"]],
    "Naive": [naive_mean["HoldingCost"], naive_mean["StockoutCost"]]
}, index=["HoldingCost", "StockoutCost"])

comparison.plot(kind="bar")
plt.title("Inventory Cost Comparison")
plt.ylabel("Cost")
plt.tight_layout()
plt.savefig("reports/figures/inventory_cost_comparison.png")
plt.close()

print("Saved inventory_cost_comparison.png")
