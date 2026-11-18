"""Bayesian Hyperparameter Optimisation for PathSnap using Optuna.

Tunes whichever model family the benchmark in src.models.train selected as
best (by default LightGBM/GradientBoosting, since boosted trees are
typically the strongest tabular candidate-classification model and the
fastest to tune meaningfully within a hackathon time budget; pass
--model to tune a different family). Uses group-aware cross-validation
(grouped by trip_id) on the 80 training trips only, as required.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_LGBM = False

from src.config import RESULTS_DIR
from src.features.extractor import FEATURE_COLUMNS
from src.models.evaluate import evaluate_predictions
from src.models.train import load_partition, subsample_per_trip

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_objective(model_family: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    def objective(trial: optuna.Trial) -> float:
        if model_family == "lightgbm" and _HAS_LGBM:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 400, step=20),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 14),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            clf = LGBMClassifier(random_state=42, verbose=-1, **params)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "max_depth": trial.suggest_int("max_depth", 6, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced", **params)

        fold_f1_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            metrics = evaluate_predictions(y_val, preds)
            fold_f1_scores.append(metrics["macro_f1"])

        return float(np.mean(fold_f1_scores))

    return objective


def optimize_model(
    model_family: str = "lightgbm" if _HAS_LGBM else "random_forest",
    n_trials: int = 30,
    n_splits: int = 5,
    max_rows_per_trip: int = 600,
) -> Dict[str, Any]:
    # Same per-trip row cap as src.models.train.run_benchmark, for the same
    # reason: 150 (trials x folds) LightGBM fits need to stay fast, and
    # 80 trips x a few hundred rows each is already ample supervision for an
    # 8-feature candidate-classification problem.
    train_df = subsample_per_trip(load_partition("train"), max_rows_per_trip)
    X_raw = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values
    groups = train_df["trip_id"].values

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    objective = _build_objective(model_family, X, y, groups, n_splits)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"pathsnap_{model_family}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - t0

    best_params = study.best_params
    if model_family == "lightgbm" and _HAS_LGBM:
        tuned_model = LGBMClassifier(random_state=42, verbose=-1, **best_params)
        untuned_model = LGBMClassifier(random_state=42, verbose=-1)
    else:
        tuned_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced", **best_params)
        untuned_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

    tuned_model.fit(X, y)
    untuned_model.fit(X, y)

    student_df = load_partition("student_test")
    X_student = scaler.transform(student_df[FEATURE_COLUMNS].values)
    y_student = student_df["label"].values

    tuned_metrics = evaluate_predictions(
        y_student, tuned_model.predict(X_student),
        y_prob=tuned_model.predict_proba(X_student) if hasattr(tuned_model, "predict_proba") else None,
    )
    untuned_metrics = evaluate_predictions(
        y_student, untuned_model.predict(X_student),
        y_prob=untuned_model.predict_proba(X_student) if hasattr(untuned_model, "predict_proba") else None,
    )

    result = {
        "model_family": model_family,
        "n_trials": n_trials,
        "best_value_cv_macro_f1": study.best_value,
        "best_params": best_params,
        "optimisation_time_s": elapsed,
        "tuned_student_test_metrics": tuned_metrics,
        "untuned_student_test_metrics": untuned_metrics,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bayesian_optimisation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return {
        "study": study,
        "scaler": scaler,
        "tuned_model": tuned_model,
        "untuned_model": untuned_model,
        **result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lightgbm" if _HAS_LGBM else "random_forest",
                         choices=["lightgbm", "random_forest"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--export", action="store_true", help="Export the tuned model/scaler for the API after optimisation.")
    args = parser.parse_args()

    out = optimize_model(model_family=args.model, n_trials=args.trials)
    print(f"Best CV Macro-F1: {out['best_value_cv_macro_f1']:.4f}")
    print(f"Best params: {out['best_params']}")
    print(f"Tuned student-test metrics: {out['tuned_student_test_metrics']}")
    print(f"Untuned student-test metrics: {out['untuned_student_test_metrics']}")
    print(f"Optimisation time: {out['optimisation_time_s']:.1f}s over {out['n_trials']} trials")

    if args.export:
        from src.models.export import export_model
        export_model(
            model=out["tuned_model"],
            scaler=out["scaler"],
            model_name=f"tuned_{args.model}",
            extra_metadata={
                "best_params": out["best_params"],
                "cv_macro_f1": out["best_value_cv_macro_f1"],
                "student_test_metrics": out["tuned_student_test_metrics"],
            },
        )
