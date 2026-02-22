import pandas as pd
import numpy as np
from . import config


# Create date features (memory safe)
def create_date_features(df):

    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")

    # Safe week extraction
    iso_week = df["date"].dt.isocalendar().week
    df["weekofyear"] = iso_week.astype("int8")

    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["quarter"] = df["date"].dt.quarter.astype("int8")

    return df


# Convert promotion to int
def process_promotion(df):

    df["onpromotion"] = df["onpromotion"].fillna(False)
    df["onpromotion"] = df["onpromotion"].astype("int8")

    return df


# Create lag features (optimized)
def create_lag_features(df):

    df = df.sort_values(["store_nbr", "item_nbr", "date"])

    grouped = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]

    for lag in config.LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag).astype("float32")

    return df


# Create rolling mean features (memory safe)
def create_rolling_features(df):

    df = df.sort_values(["store_nbr", "item_nbr", "date"])

    grouped = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]

    for window in config.ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            grouped.shift(1)
            .rolling(window)
            .mean()
            .astype("float32")
        )

    return df


def feature_pipeline(train, test):

    # Date features
    train = create_date_features(train)
    test = create_date_features(test)

    # Promotion processing
    train = process_promotion(train)
    test = process_promotion(test)

    # Sort once
    train = train.sort_values(["store_nbr", "item_nbr", "date"])

    # Create lag + rolling ONLY for train
    train = create_lag_features(train)
    train = create_rolling_features(train)

    return train, test