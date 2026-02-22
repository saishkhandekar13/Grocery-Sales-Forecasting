import pandas as pd
import numpy as np


# Clean target variable
def clean_target(train):

    train["unit_sales"] = train["unit_sales"].clip(lower=0)
    train["unit_sales"] = np.log1p(train["unit_sales"])

    return train


# Merge all external datasets
def merge_datasets(train, test, stores, items, holidays, oil):

    stores = stores.drop_duplicates(subset=["store_nbr"])
    items = items.drop_duplicates(subset=["item_nbr"])
    oil = oil.drop_duplicates(subset=["date"])

    train = train.merge(stores, on="store_nbr", how="left")

    train = train.merge(items, on="item_nbr", how="left")

    train = train.merge(oil, on="date", how="left")

    holiday_dates = set(holidays["date"].unique())
    train["is_holiday"] = train["date"].isin(holiday_dates).astype("int8")

    return train, test