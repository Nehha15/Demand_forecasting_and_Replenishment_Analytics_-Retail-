# 🛒 Retail Demand Forecasting & Replenishment Analytics

## 📌 Overview

Built an end-to-end demand forecasting and inventory optimisation pipeline inspired by UK grocery retail operations.

The objective was not only to improve forecast accuracy, but to quantify operational impact on:

- Stock availability  
- Stockout cost  
- Inventory holding cost  
- Service level  

---

## 🎯 Business Problem

In grocery retail, inaccurate forecasts lead to:

- Lost sales during peak trading  
- Excess working capital tied in inventory  
- Reduced on-shelf availability  

This project evaluates whether machine learning can outperform standard retail heuristics and deliver measurable inventory improvements.

---

## 🧠 Approach

### 1️⃣ Time-Series Feature Engineering

- Lag features (1, 2, 4, 12, 52 weeks)
- Rolling means and volatility
- Calendar features
- Holiday and pre-holiday indicators
- Store metadata and economic signals

A chronological train-validation split was used to prevent data leakage.

---

### 2️⃣ Benchmarking Against Retail Heuristics

Compared ML models against:

- Last-week sales (lag_1)
- Same-week-last-year (lag_52)
- Rolling average

| Model | MAE | Improvement |
|-------|------|-------------|
| Naïve | 1751 | — |
| RandomForest | 1395 | ~20% |

Holiday MAE improved by ~25%.

---

### 3️⃣ Interpretability (SHAP)

SHAP analysis showed:

- Annual seasonality (lag_52) dominates demand.
- Short-term momentum (lag_1) is secondary.
- Promotional markdown variables had limited marginal impact.

This confirms strong recurring demand cycles aligned to retail calendar events.

---

## 📦 Replenishment Simulation

To connect forecasting with operations, a weekly **order-up-to policy** was simulated.

### Results (Validation Period)

| Metric | RandomForest | Naïve |
|--------|-------------|--------|
| Holding Cost | 189K | 191K |
| Stockout Cost | 9K | 19K |
| Service Level | 99.31% | 98.94% |

Machine learning reduced stockout cost by ~54% while maintaining similar holding cost.

---

## 🏗 Tech Stack

Python · Pandas · scikit-learn · SHAP · Matplotlib

Modular structure separating:

- Data processing  
- Feature engineering  
- Model training  
- Evaluation  
- Inventory simulation  

---

## 🚀 Key Takeaway

Improved forecast accuracy translated into materially lower stockout cost and higher service level — demonstrating how data science can directly support availability and supply chain decision-making in grocery retail.
