import tempfile
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def train_and_log(model_name, model, X_train, X_test, y_train, y_test) -> dict:
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        }
        loggable = {k: v for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(loggable)

        mlflow.sklearn.log_model(model, artifact_path="model")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(model_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "confusion_matrix.png")
            fig.savefig(path, bbox_inches="tight")
            mlflow.log_artifact(path)
        plt.close(fig)

    return {"model": model_name, **metrics}
