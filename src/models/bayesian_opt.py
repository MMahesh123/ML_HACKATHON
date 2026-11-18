"""Bayesian Hyperparameter Optimization for PathSnap using Optuna."""

from typing import Dict, Any
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from src.models.evaluate import evaluate_predictions

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_random_forest(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int = 30) -> Dict[str, Any]:
    """
    Performs Bayesian optimization on Random Forest hyperparameters
    using Group-aware cross-validation on the 80 training trips.
    """
    gkf = GroupKFold(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 250, step=25)
        max_depth = trial.suggest_int("max_depth", 6, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

        fold_f1_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            metrics = evaluate_predictions(y_val, preds)
            fold_f1_scores.append(metrics["macro_f1"])

        return float(np.mean(fold_f1_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study": study,
    }


if __name__ == "__main__":
    print("PathSnap Bayesian Hyperparameter Optimization module ready.")
