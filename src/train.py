import pandas as pd
from datetime import timedelta
from . import config
from .model import (
    get_linear_model,
    get_ridge_model,
    get_random_forest,
    get_xgboost
)
from .evaluate import evaluate_model


# Time-based split
def time_split(data):

    max_date = data["date"].max()
    split_date = max_date - timedelta(days=config.VALIDATION_DAYS)

    train_data = data[data["date"] < split_date]
    valid_data = data[data["date"] >= split_date]

    return train_data, valid_data


# Prepare features and target
def prepare_data(df):

    # Drop rows with NaN (from lag features)
    df = df.dropna().copy()

    # Separate target
    y = df["unit_sales"]

    # Drop non-feature columns
    X = df.drop(columns=["unit_sales"])

    # Drop date column
    if "date" in X.columns:
        X = X.drop(columns=["date"])

    # Convert any remaining object columns to numeric
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Convert category columns too
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].cat.codes

    return X, y


# Train and evaluate all models
def train_models(data):

    train_data, valid_data = time_split(data)

    X_train, y_train = prepare_data(train_data)
    X_valid, y_valid = prepare_data(valid_data)

    models = {
        "LinearRegression": get_linear_model(),
        "Ridge": get_ridge_model(),
        "RandomForest": get_random_forest(),
        "XGBoost": get_xgboost()
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_train, y_train,
            X_valid, y_valid
        )

        results[name] = metrics

        print(f"{name} Results:")
        print(metrics)

    return results