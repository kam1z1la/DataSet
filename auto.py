import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

def get_dataset():
    df = pd.read_csv(
        DATASET_URL,
        names=headers
    )

    return df

def replace_missing_values(df):
    df.replace("?", np.nan, inplace=True)

    return df

def replace_stroke_missing_values(df):
    df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

    mean_stroke = df["stroke"].mean()

    print(f"Mean stroke: {mean_stroke}")

    df["stroke"] = df["stroke"].fillna(mean_stroke)

    return df

def normalize_height(df):
    df["height"] = pd.to_numeric(df["height"], errors="coerce")

    min_height = df["height"].min()
    max_height = df["height"].max()

    print(f"Height data type: {df['height'].dtype}")
    print(f"Minimum height: {min_height}")
    print(f"Maximum height: {max_height}")

    df["height"] = (
        (df["height"] - min_height)
        / (max_height - min_height)
    )

    return df

def create_aspiration_indicator(df):
    indicator = pd.get_dummies(
        df["aspiration"],
        prefix="aspiration"
    )

    return indicator

def merge_aspiration_indicator(df, indicator):
    df = pd.concat(
        [df, indicator],
        axis=1
    )

    return df

def remove_aspiration(df):
    df.drop("aspiration", axis=1, inplace=True)

    return df

def check_missing_values(df):
    missing_values = df.isna().sum()

    print("Missing values in each column:")
    print(missing_values[missing_values > 0])

    total_missing = missing_values.sum()

    print(f"\nTotal missing values: {total_missing}")

    return total_missing == 0


def main():
    df = get_dataset()

    print("Old dataset:")
    print(df.head())

    df = replace_missing_values(df)


    print("\nStroke before replacement:")
    print(df["stroke"].head(10))

    df = replace_stroke_missing_values(df)

    print("\nStroke after replacement:")
    print(df["stroke"].head(10))

    print("\nMissing values in stroke:")
    print(df["stroke"].isna().sum())

    print("\n New Dataset:")
    print(df.head())

    dh = normalize_height(df)

    print(dh["height"].head())


    no_missing_values = check_missing_values(df)

    if no_missing_values:
        print("\nAll missing values have been replaced.")
    else:
        print("\nSome missing values remain.")

if __name__ == "__main__":
    main()