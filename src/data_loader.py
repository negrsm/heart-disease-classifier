import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
