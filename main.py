from src.data_loader import load_raw_data, merge_data, clean_data
from src.features import create_time_features,create_lag_features,create_rolling_features,finalize_features


DATA_PATH = "data/raw"
# OUTPUT_PATH = "data/processed/clean_sales.csv"
OUTPUT_PATH = "data/processed/model_ready.csv"

def main():
    train, features, stores = load_raw_data(DATA_PATH)
    df = merge_data(train, features, stores)
    df = clean_data(df)

    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = finalize_features(df)
    

    df.to_csv(OUTPUT_PATH, index=False)
    print("Clean dataset saved successfully.")


if __name__ == "__main__":
    main()
