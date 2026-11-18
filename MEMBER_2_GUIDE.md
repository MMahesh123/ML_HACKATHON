# PathSnap — Member 2: 5-Model Benchmarking & Optuna Bayesian Optimization

## Role & Objectives
You own **Phase 5 through Phase 7** (Workflow Stages 5–7, Kaggle Playbook sections 08–10):
1. **5 Baseline Models:** Implement pipelines for 5 distinct algorithm families:
   - Logistic Regression (Linear / Probabilistic baseline + `StandardScaler`)
   - k-Nearest Neighbors (`weights='distance'`, `n_neighbors=15`)
   - Decision Tree (`max_depth=12`, `min_samples_leaf=10`)
   - Random Forest (`n_estimators=300`, `max_depth=18`)
   - HistGradientBoosting / LightGBM (Sequential boosting)
2. **Group-Aware Cross-Validation:** Run 5-fold `GroupKFold` split by `trip_id` on the 80 training trips. Candidates from the same trip must never appear in both train and validation folds.
3. **Student-Test Evaluation:** Evaluate the 5 frozen baseline models on the 10 student-test trips.
4. **Bayesian Optimization:** Use **Optuna** to tune the winning model (e.g. HistGradientBoosting) on an 80-trip GroupKFold cross-validation objective.
5. **Model Export:** Save `candidate_ranker.joblib`, scalers, and `metadata.json`.

## Evaluation Metrics to Report:
- Candidate-level Binary Precision, Recall, F1
- Observation-level Top-1 Accuracy
- Log Loss (probability calibration)
- Training runtime & inference latency per 1000 points
