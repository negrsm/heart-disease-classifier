# classification-mlflow

A multi-model classification comparison project using MLflow for experiment tracking.

## Overview

This project trains and compares multiple classification models on a dataset, logging all experiments — parameters, metrics, and artifacts — to MLflow for side-by-side comparison.

### Models compared

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Metrics tracked

- Accuracy
- Weighted F1 Score
- ROC-AUC

## Project structure

```
classification-mlflow/
├── data/           # Raw and processed data (not tracked by git)
├── notebooks/      # Exploratory notebooks
├── src/            # Source modules
├── main.py         # Entry point: trains all models and logs to MLflow
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Implement `load_data()` in [main.py](main.py) to return `(X, y)` arrays.
2. Run the experiment:

```bash
python main.py
```

3. Launch the MLflow UI to compare runs:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.
