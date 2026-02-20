import pandas as pd
import numpy as np


def create_time_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsPreHoliday"] = df.groupby(["Store", "Dept"])["IsHoliday"].shift(-1).fillna(0).astype(int)

    return df


def create_lag_features(df, lags=[1, 2, 4, 12, 52]):
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(lag)
        )
    return df


def create_rolling_features(df, windows=[4, 8]):
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(1)  # Prevent leakage
            .rolling(window)
            .mean()
        )

        df[f"rolling_std_{window}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(1)
            .rolling(window)
            .std()
        )

    return df


def finalize_features(df):
    # Drop rows with NA caused by lagging
    df = df.dropna()

    return df
