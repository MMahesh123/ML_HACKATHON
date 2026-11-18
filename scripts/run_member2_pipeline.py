"""
Complete execution runner for Member 2:
- Phase 5: 5-Model GroupKFold Cross-Validation on training trips
- Phase 6: Frozen Student-Test Benchmarking across all 5 models
- Phase 7: Optuna Bayesian Hyperparameter Optimization on the winning model
- Model & Metadata Artifact Export
"""

import time
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR
from src.models.train import run_group_kfold_cv, evaluate_student_test, FEATURE_COLS
from src.models.bayesian_opt import run_bayesian_optimization


def main():
    start_all = time.perf_counter()
    print("=" * 70)
    print("PathSnap — Member 2: Model Benchmarking & Bayesian Optimization Pipeline")
    print("=" * 70)

    train_file = PROCESSED_DATA_DIR / "train_candidates.parquet"
    test_file = PROCESSED_DATA_DIR / "student_test_candidates.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Candidate datasets not found. Please run feature extraction first."
        )

    print(f"Loading training candidate features from {train_file.name}...")
    df_train = pd.read_parquet(train_file)
    print(f"Loaded train candidates: {df_train.shape} across {df_train['trip_id'].nunique()} trips.")

    print(f"Loading student_test candidate features from {test_file.name}...")
    df_test = pd.read_parquet(test_file)
    print(f"Loaded test candidates: {df_test.shape} across {df_test['trip_id'].nunique()} trips.")

    # -------------------------------------------------------------
    # 1. 5-Model GroupKFold Cross-Validation (Phase 5)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: 5-Model GroupKFold Cross-Validation")
    print("=" * 70)
    df_folds, df_summary = run_group_kfold_cv(df_train, feature_cols=FEATURE_COLS, n_splits=5)

    print("\n--- Cross-Validation Summary Across 5 Algorithm Families ---")
    print(df_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Frozen Student-Test Evaluation (Phase 6)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: Baseline Evaluation on Frozen Student-Testing Trips")
    print("=" * 70)
    df_test_res = evaluate_student_test(df_train, df_test, feature_cols=FEATURE_COLS)

    print("\n--- Student-Test Performance Summary ---")
    print(df_test_res.to_string(index=False))

    # -------------------------------------------------------------
    # 3. Bayesian Hyperparameter Optimization with Optuna (Phase 7)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 7: Optuna Bayesian Hyperparameter Optimization")
    print("=" * 70)
    opt_results = run_bayesian_optimization(
        df_train=df_train,
        df_test=df_test,
        feature_cols=FEATURE_COLS,
        n_trials=20,
    )

    total_duration = time.perf_counter() - start_all
    print("\n" + "=" * 70)
    print(f"ALL MEMBER 2 PHASES COMPLETED IN {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Generated Artifacts:")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"  - Model: {MODELS_DIR / 'candidate_ranker.joblib'}")
    print(f"  - Metadata: {MODELS_DIR / 'metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
