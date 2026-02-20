import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str):
    data_path = Path(data_path)

    train = pd.read_csv(data_path / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(data_path / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_path / "stores.csv")

    return train, features, stores


def merge_data(train, features, stores):
    # Merge train with features
    df = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    # Merge store metadata
    df = df.merge(
        stores,
        on="Store",
        how="left"
    )

    return df

def clean_data(df):
    # Fill MarkDown columns with 0 (promotion not active)
    markdown_cols = [col for col in df.columns if "MarkDown" in col]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # Forward fill CPI and Unemployment per store
    df["CPI"] = df.groupby("Store")["CPI"].transform(lambda x: x.ffill())
    df["Unemployment"] = df.groupby("Store")["Unemployment"].transform(lambda x: x.ffill())

    # Sort properly (CRITICAL for time-series features later)
    df = df.sort_values(["Store", "Dept", "Date"])

    return df

