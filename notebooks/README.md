# Jupyter Notebooks Guide

The notebooks in this directory follow the required experimental workflow specified in the PathSnap Hackathon Guidelines:

1. **`01_eda_and_exploration.ipynb`**:
   - Trajectory length distributions, sampling interval consistency (~1Hz checks).
   - Missing/invalid observation scans.
   - Road network connectivity analysis (isolated nodes, graph components).
   - Baseline Folium geographical maps showing sample trips overlaid on the road network.

2. **`02_candidate_generation.ipynb`**:
   - Spatial indexing (k-d tree) for querying nearest road arcs.
   - Feature engineering: perpendicular distance, heading delta ($\Delta \theta$), projection ratio $t$, trajectory speed.
   - Construction of observation-candidate training pairs.

3. **`03_model_experimentation.ipynb`**:
   - Training and benchmarking at least **5 distinct models** from different algorithm families (Linear/Probabilistic, k-NN, Decision Tree, Random Forest, LightGBM/XGBoost).
   - Group-K-Fold cross-validation (grouped by `trip_id`) on the 80 training trips.
   - Comparison on 10 student-testing trips (Top-1 Accuracy, Macro F1, Log Loss, inference time).

4. **`04_bayesian_optimization.ipynb`**:
   - Optuna Bayesian optimization search on the chosen model family.
   - Objective: Cross-validation Macro F1 score on training trips.
   - Comparison of tuned vs. untuned model metrics.

5. **`05_hmm_viterbi_evaluation.ipynb`**:
   - Formulating emission probabilities (classifier output + distance error).
   - Formulating transition probabilities (shortest path network distance vs. Euclidean displacement).
   - Viterbi MAP trajectory decoding and comparison: classifier-only vs. classifier + HMM.
   - Analysis of route continuity and topological recovery.
