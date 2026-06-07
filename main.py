import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_data():
    # Replace with actual data loading logic
    raise NotImplementedError("Implement data loading in load_data()")


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="weighted"),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    return metrics


def run_experiment(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"{name}: {metrics}")


def main():
    mlflow.set_experiment("classification-comparison")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    for name, model in models.items():
        run_experiment(name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
