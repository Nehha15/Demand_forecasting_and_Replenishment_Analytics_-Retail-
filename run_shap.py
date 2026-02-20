import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor

# Ensure output directory exists
os.makedirs("reports/figures", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/model_ready.csv", parse_dates=["Date"])
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

TARGET = "Weekly_Sales"
FEATURES = [col for col in df.columns if col not in ["Weekly_Sales", "Date"]]

train_idx = df["Date"] < "2012-01-01"

train = df[train_idx]
val = df[~train_idx]

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = val[FEATURES]

# Train model
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# Use smaller sample for speed
sample = X_val.sample(1000, random_state=42)

shap_values = explainer.shap_values(sample)

# Summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()
plt.savefig("reports/figures/shap_summary.png")
plt.close()

print("Saved shap_summary.png")
