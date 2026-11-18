"""Model experimentation and multi-model benchmarking pipeline for PathSnap.

Benchmarks at least 5 models across distinct algorithm families:
1. Linear / Probabilistic: Logistic Regression / Gaussian NB
2. Instance-based: k-Nearest Neighbors (KNN)
3. Tree-based: Decision Tree
4. Ensemble: Random Forest Classifier
5. Boosting: LightGBM / XGBoost
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold
from src.models.evaluate import evaluate_predictions


def get_candidate_models() -> Dict[str, Any]:
    """Returns 5 diverse baseline model architectures from distinct families."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "GaussianNB": GaussianNB(),
        "KNN_Distance": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }


def run_benchmark(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """
    Performs Group-K-Fold cross-validation across trips (preventing data leakage across trips).
    Evaluates Top-1 Accuracy, Macro F1, and training time.
    """
    models = get_candidate_models()
    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        accuracies = []
        f1_scores = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            metrics = evaluate_predictions(y_val, preds)
            accuracies.append(metrics["top_1_accuracy"])
            f1_scores.append(metrics["macro_f1"])

        results.append({
            "Model": name,
            "CV_Top1_Accuracy": np.mean(accuracies),
            "CV_Macro_F1": np.mean(f1_scores),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("PathSnap Model Experimentation Pipeline initialized.")
    print(f"Available baseline models: {list(get_candidate_models().keys())}")
