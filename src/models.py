from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }


def get_tuned_model() -> GridSearchCV:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=42)
    return GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1)
