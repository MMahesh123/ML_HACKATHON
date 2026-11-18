"""Orchestrates the full PathSnap offline pipeline end to end:

1. Build the candidate-classification dataset from raw trip files.
2. Benchmark 6 models with group-aware CV + student-test + sequence-level metrics.
3. Bayesian-optimise the strongest model family.
4. Export the final tuned model + scaler for the FastAPI backend to load.

Run with: python -m scripts.run_full_pipeline
"""

from __future__ import annotations

import pandas as pd

from src.config import STUDENT_TEST_DIR, TRAIN_DIR
from src.data.build_dataset import build_partition
from src.models.bayesian_opt import optimize_model
from src.models.export import export_model
from src.models.train import run_benchmark


def main():
    print("\n### Step 1/4: Building candidate-classification dataset ###")
    build_partition(TRAIN_DIR, "train")
    build_partition(STUDENT_TEST_DIR, "student_test")

    print("\n### Step 2/4: Benchmarking 6 models ###")
    comparison_df = run_benchmark()
    best_model_name = comparison_df.sort_values("CV_Macro_F1", ascending=False).iloc[0]["Model"]
    print(f"\nBest model by CV Macro-F1: {best_model_name}")

    print("\n### Step 3/4: Bayesian hyperparameter optimisation ###")
    family = "lightgbm" if "LightGBM" in best_model_name or "Gradient" in best_model_name else "random_forest"
    opt_result = optimize_model(model_family=family, n_trials=30)

    print("\n### Step 4/4: Exporting final tuned model ###")
    export_model(
        model=opt_result["tuned_model"],
        scaler=opt_result["scaler"],
        model_name=f"tuned_{family}",
        extra_metadata={
            "best_params": opt_result["best_params"],
            "cv_macro_f1": opt_result["best_value_cv_macro_f1"],
            "student_test_metrics": opt_result["tuned_student_test_metrics"],
        },
    )
    print("\nPipeline complete. Start the API with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
