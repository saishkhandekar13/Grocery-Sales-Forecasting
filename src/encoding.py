import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_features(train, test):

    categorical_cols = train.select_dtypes(include=["object", "category"]).columns

    encoders = {}

    for col in categorical_cols:

        if col not in test.columns:
            continue

        le = LabelEncoder()

        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

        le.fit(train[col])

        train[col] = le.transform(train[col])

        test[col] = test[col].map(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )

        encoders[col] = le

    return train, test, encoders