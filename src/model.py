from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from . import config


# Linear Regression
def get_linear_model():
    return LinearRegression()


# Ridge Regression
def get_ridge_model():
    return Ridge(alpha=1.0, random_state=config.RANDOM_STATE)


# Random Forest (balanced speed + strength)
def get_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )


# XGBoost (main strong model)
def get_xgboost():
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",   # faster for large datasets
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )