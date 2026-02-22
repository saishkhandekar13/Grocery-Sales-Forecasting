import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# Evaluate model
def evaluate_model(model, X_train, y_train, X_valid, y_valid):

    # Training predictions
    train_preds = model.predict(X_train)

    # Validation predictions
    valid_preds = model.predict(X_valid)

    # Reverse log transform
    train_preds = np.expm1(train_preds)
    valid_preds = np.expm1(valid_preds)

    y_train_actual = np.expm1(y_train)
    y_valid_actual = np.expm1(y_valid)

    results = {}

    # Training metrics
    results["train_mae"] = mean_absolute_error(y_train_actual, train_preds)
    results["train_rmse"] = np.sqrt(mean_squared_error(y_train_actual, train_preds))
    results["train_rmsle"] = rmsle(y_train_actual, train_preds)

    # Validation metrics
    results["valid_mae"] = mean_absolute_error(y_valid_actual, valid_preds)
    results["valid_rmse"] = np.sqrt(mean_squared_error(y_valid_actual, valid_preds))
    results["valid_rmsle"] = rmsle(y_valid_actual, valid_preds)

    return results