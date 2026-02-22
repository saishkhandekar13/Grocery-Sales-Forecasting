import pandas as pd
from . import config


def load_data():

    # Load limited rows for memory safety
    train = pd.read_csv(
        config.TRAIN_FILE,
        nrows=5000000,
        dtype={
            "store_nbr": "int16",
            "item_nbr": "int32",
            "unit_sales": "float32",
        },
        low_memory=False
    )

    train["date"] = pd.to_datetime(train["date"])

    # Load test (small)
    test = pd.read_csv(
        config.TEST_FILE,
        dtype={
            "store_nbr": "int16",
            "item_nbr": "int32",
        }
    )
    test["date"] = pd.to_datetime(test["date"])

    # Small datasets
    stores = pd.read_csv(config.STORES_FILE)
    items = pd.read_csv(config.ITEMS_FILE)
    holidays = pd.read_csv(config.HOLIDAYS_FILE)
    oil = pd.read_csv(config.OIL_FILE)

    holidays["date"] = pd.to_datetime(holidays["date"])
    oil["date"] = pd.to_datetime(oil["date"])

    return train, test, stores, items, holidays, oil