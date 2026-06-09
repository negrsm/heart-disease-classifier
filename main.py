import pandas as pd
import mlflow

from src.data_loader import load_data, split_data
from src.models import get_models
from src.train import train_and_log


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("classification-comparison")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    results = []
    for name, model in get_models().items():
        result = train_and_log(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    summary = (
        pd.DataFrame(results)
        .set_index("model")
        .sort_values("f1", ascending=False)
    )
    print("\n=== Results (sorted by F1) ===")
    print(summary.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
