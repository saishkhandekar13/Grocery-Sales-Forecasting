from src.data_loader import load_data
from src.preprocessing import clean_target, merge_datasets
from src.feature_engineering import feature_pipeline
from src.encoding import encode_features
from src.train import train_models


def main():

    # Load data
    train, test, stores, items, holidays, oil = load_data()

    # Clean target
    train = clean_target(train)

    # Merge datasets
    train, test = merge_datasets(
        train, test, stores, items, holidays, oil
    )

    print("Rows after merge:", len(train))

    # Feature engineering
    train, test = feature_pipeline(train, test)

    # Encode categorical features
    train, test, encoders = encode_features(train, test)

    # Train and evaluate models
    results = train_models(train)

    print("\nFinal Model Comparison:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()