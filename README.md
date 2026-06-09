# Heart Disease Classifier

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-189FDD)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

## Description

A systematic comparison of four classification models on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland, 303 patients), tracking every experiment with MLflow. The pipeline loads and cleans the data, trains Logistic Regression, Random Forest, a GridSearchCV-tuned Random Forest, and XGBoost, then logs parameters, five metrics, trained model artifacts, and confusion matrix plots for each run — all queryable from the MLflow UI or the included Jupyter notebook. The goal is to demonstrate end-to-end ML experiment tracking as it would be done in a production data science workflow: reproducible runs, a central artifact store, and a clean comparison surface.

## Results

> Metrics from a stratified 80/20 split (`random_state=42`) on 297 samples (6 dropped for missing values). Best value per column in **bold**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** ★ | **0.8814** | 0.8929 | 0.8929 | **0.8929** | **0.9495** |
| Random Forest (Tuned) | 0.8644 | **0.8966** | 0.8571 | 0.8763 | 0.9444 |
| Random Forest | 0.8644 | 0.8621 | **0.8929** | 0.8772 | 0.9327 |
| Logistic Regression | 0.8475 | 0.8571 | 0.8571 | 0.8571 | 0.9167 |

## Project Structure

```
classification-mlflow/
├── notebooks/
│   └── model_comparison.ipynb   # metrics table, bar chart, confusion matrices
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # load & split UCI Heart Disease data
│   ├── models.py                # model definitions + GridSearchCV tuned RF
│   └── train.py                 # MLflow run: fit, evaluate, log artifacts
├── main.py                      # entry point — trains all 4 models
├── requirements.txt
└── README.md
```

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/negrsm/heart-disease-classifier.git
cd heart-disease-classifier
```

**2. Create and activate the conda environment**
```bash
conda create -n classification-mlflow python=3.10 -y
conda activate classification-mlflow
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train all models and log to MLflow**
```bash
python main.py
```

The script prints a results summary sorted by F1 score when it finishes.

**5. Explore runs in the MLflow UI**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` — compare metrics, inspect parameters, and download confusion matrix artifacts.

**6. Open the comparison notebook** *(optional)*
```bash
jupyter notebook notebooks/model_comparison.ipynb
```

Run all cells to see the styled metrics table, grouped bar chart, and side-by-side confusion matrices.

## Key Findings

- **XGBoost is the overall winner**, achieving the highest F1 (0.893) and ROC-AUC (0.950) — it handles the non-linear feature interactions (e.g. `thal`, `cp`, `ca`) that penalise the linear model.
- **GridSearchCV tuning boosted Random Forest's precision to 0.897**, the highest of all models, by constraining tree depth and raising the split threshold — a useful trade-off when the cost of false positives is high.
- **All four models exceed 0.91 ROC-AUC**, confirming the 13 clinical features are strongly discriminative; even the logistic regression baseline is competitive.
- **Logistic Regression sits within 4 F1 points of XGBoost** while being fully interpretable — a reasonable production choice if explainability is a hard requirement.

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

## License

This project is licensed under the [MIT License](LICENSE).
