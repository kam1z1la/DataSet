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

def normalize_height(df):
    min_height = df["height"].min()
    max_height = df["height"].max()

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

def main():
    df = get_dataset()

    print("Old dataset:")
    print(df.head())

    df = replace_missing_values(df)

    print("\n New Dataset:")
    print(df.head())

    dh = normalize_height(df)

    print(dh["height"].head())

if __name__ == "__main__":
    main()