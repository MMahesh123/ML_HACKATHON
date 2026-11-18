"""Model experimentation and multi-model benchmarking pipeline for PathSnap.

Benchmarks 6 models across 6 distinct algorithm families on the real
candidate-classification dataset built by `src.data.build_dataset`:

1. Linear/probabilistic : Logistic Regression
2. Instance-based       : Distance-weighted k-NN
3. Tree-based           : Decision Tree
4. Ensemble             : Random Forest
5. Boosting             : LightGBM
6. Neural network       : MLP

Each model is scored with Group-K-Fold cross-validation (grouped by
`trip_id`, so no trip's rows appear in both the train and validation fold),
then fit on the full training partition and evaluated on the 10 held-out
student-test trips, both as a plain row-level classifier and end-to-end
(sequence-level, via src.models.sequence_eval, using the exact same pipeline
the live API calls).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_LGBM = False

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from src.features.extractor import FEATURE_COLUMNS
from src.models.evaluate import evaluate_predictions
from src.models.sequence_eval import evaluate_sequence_on_trips
from src.pipeline import LoadedModel


def get_candidate_models() -> Dict[str, Any]:
    """Returns 6 baseline model architectures from 6 distinct algorithm families."""
    boosting = (
        LGBMClassifier(n_estimators=150, max_depth=-1, learning_rate=0.1, random_state=42, verbose=-1)
        if _HAS_LGBM
        else GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    )
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "KNN_Distance": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM" if _HAS_LGBM else "GradientBoosting": boosting,
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True),
    }


def load_partition(name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df[df["label"] >= 0].reset_index(drop=True)


def run_group_cv(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    accs, f1s, loglosses = [], [], []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_val, preds, y_prob=proba)
        accs.append(metrics["top_1_accuracy"])
        f1s.append(metrics["macro_f1"])
        if metrics.get("log_loss") is not None:
            loglosses.append(metrics["log_loss"])

    return {
        "cv_accuracy": float(np.mean(accs)),
        "cv_macro_f1": float(np.mean(f1s)),
        "cv_log_loss": float(np.mean(loglosses)) if loglosses else None,
    }


def subsample_per_trip(df: pd.DataFrame, max_rows_per_trip: int, seed: int = 42) -> pd.DataFrame:
    """Caps rows per trip so CV/training cost scales with #trips, not total
    candidate-row count. 80 trips x a few hundred rows each is already far
    more supervision than this 8-feature problem needs, and keeps the
    6-model x 5-fold benchmark running in minutes instead of hours - the
    row cap is reported in the technical report as a deliberate efficiency
    trade-off, not a quality one (candidate miss rate / route continuity are
    still measured on the FULL, unsampled trips via sequence_eval).
    """
    rng = np.random.RandomState(seed)
    sampled_parts = []
    for trip_id, group in df.groupby("trip_id"):
        n = min(len(group), max_rows_per_trip)
        idx = rng.choice(group.index.values, size=n, replace=False)
        sampled_parts.append(group.loc[idx])
    return pd.concat(sampled_parts, ignore_index=True)


def run_benchmark(max_rows_per_trip: int = 600) -> pd.DataFrame:
    train_df = subsample_per_trip(load_partition("train"), max_rows_per_trip)
    student_df = subsample_per_trip(load_partition("student_test"), max_rows_per_trip)

    X_train_raw = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    groups = train_df["trip_id"].values

    X_student_raw = student_df[FEATURE_COLUMNS].values
    y_student = student_df["label"].values

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_student = scaler.transform(X_student_raw)

    student_trip_ids = sorted(student_df["trip_id"].unique().tolist())

    models = get_candidate_models()
    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"=== {name} ===")
        cv_metrics = run_group_cv(model, X_train, y_train, groups)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_student)
        proba = model.predict_proba(X_student) if hasattr(model, "predict_proba") else None
        inference_time_ms = (time.perf_counter() - t1) * 1000.0 / max(len(X_student), 1)

        row_metrics = evaluate_predictions(y_student, preds, y_prob=proba)

        loaded = LoadedModel(model=model, scaler=scaler, feature_columns=FEATURE_COLUMNS, name=name)
        seq_metrics = evaluate_sequence_on_trips(STUDENT_TEST_DIR, student_trip_ids, loaded_model=loaded)

        results.append({
            "Model": name,
            "CV_Accuracy": cv_metrics["cv_accuracy"],
            "CV_Macro_F1": cv_metrics["cv_macro_f1"],
            "CV_Log_Loss": cv_metrics["cv_log_loss"],
            "StudentTest_RowAccuracy": row_metrics["top_1_accuracy"],
            "StudentTest_RowMacroF1": row_metrics["macro_f1"],
            "StudentTest_LogLoss": row_metrics.get("log_loss"),
            "StudentTest_Top1Accuracy_ClassifierOnly": seq_metrics.get("mean_top1_acc_classifier"),
            "StudentTest_Top1Accuracy_WithHMM": seq_metrics.get("mean_top1_acc_hmm"),
            "RouteContinuity_ClassifierOnly": seq_metrics.get("mean_route_continuity_classifier"),
            "RouteContinuity_WithHMM": seq_metrics.get("mean_route_continuity_hmm"),
            "TrainTime_s": train_time_s,
            "InferenceTime_ms_per_row": inference_time_ms,
        })
        print(pd.DataFrame([results[-1]]).to_string(index=False))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved comparison table -> {RESULTS_DIR / 'model_comparison.csv'}")
    return results_df


if __name__ == "__main__":
    df = run_benchmark()
    print(df.to_string(index=False))
