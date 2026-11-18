# Team Contributions — PathSnap

PES University · UE24CS352A Machine Learning Hackathon
Team of 4 members.

---

## Member 1 — Geospatial Data Engineering, EDA & Sequence Alignment Gate

**Primary Environment:** Kaggle Notebook
**Notebook Cells Owned:** 00 — Setup through 07 — Features
**Workflow Stages:** Stage 1 (Raw Dataset), Stage 2 (EDA), Stage 3 (Preprocessing), Stage 4 (Candidates & Features)

**Detailed Responsibilities:**

- **Data Ingestion & Integrity Checks (Cells 00–02)**
  - Load `.track`, `.nodes`, `.arcs`, and `.route` across the 80 train and 10 student-test trips.
  - Generate `dataset_inventory.csv` (trip IDs, row counts, timestamp monotonicity, missing coordinates).
- **Exploratory Data Analysis & Geometry (Cells 03–04)**
  - Project angular coordinates into metric planar coordinates (or compute geodesics); never use raw lat/lon Euclidean distance as meters.
  - Generate Folium HTML maps visualizing raw GPS trajectories against road network arcs.
- **Candidate Generation (Cell 05)**
  - Implement spatial search (KD-Tree within 50m) to generate candidate road segments for each GPS point.
  - Log candidate coverage and candidate miss rate (ensuring reference arcs are not omitted).
- **MANDATORY GATE — Sequence Alignment (Cell 06)**
  - Crucial Rule: `.track` (e.g., 1,315 rows) and `.route` (e.g., 525 arcs) do not have 1:1 row correspondence.
  - Design and validate an order-preserving dynamic programming / monotonic sequence alignment to map each GPS observation to its reference route arc before creating labels.
- **Feature Matrix Construction (Cell 07)**
  - Extract candidate features: `point_segment_distance_m`, `heading_difference_deg`, `gps_speed_mps`, `segment_length_m`, `segment_bearing_deg`, `time_gap_s`, and `candidate_rank_distance`.
  - Formulate `(X, y, groups)` where `y ∈ {0, 1}` and `groups = trip_id`.

**Maps to code:** `src/data/loader.py`, `src/data/preprocessor.py`, `src/features/candidate.py`, `src/features/extractor.py`, `src/data/build_dataset.py`, `scripts/run_eda.py`, `artifacts/results/dataset_inventory.csv`.

---

## Member 2 — Five ML Models Benchmarking & Bayesian Optimization

**Primary Environment:** Kaggle Notebook
**Notebook Cells Owned:** 08 — CV, 09 — Student test, 10 — Optimisation
**Workflow Stages:** Stage 5 (Five ML Models), Stage 6 (Model Selection), Stage 7 (Bayesian Optimisation)

**Detailed Responsibilities:**

- **Model Pipelines Implementation (Cell 08)**
  - Set up `Pipeline([('imputer', SimpleImputer()), ('scale', StandardScaler()), ('model', ...)])` for the 5 distinct families:
    - Logistic Regression (linear/probabilistic baseline, `class_weight='balanced'`)
    - k-Nearest Neighbors (instance-based, `weights='distance'`, `n_neighbors=15`)
    - Decision Tree (tree-based, `max_depth=12`, `min_samples_leaf=10`)
    - Random Forest (ensemble bagging, `n_estimators=300`, `max_depth=18`)
    - HistGradientBoosting / LightGBM (sequential tree boosting)
- **Group-Aware Cross-Validation (Cell 08)**
  - Run 5-fold `GroupKFold` split strictly by `trip_id` on the 80 training trips (preventing leakage across folds).
  - Record candidate-level binary F1, observation-level Top-1 Accuracy, training time, and inference latency.
  - Save `model_cv_folds.csv` and `model_cv_summary.csv`.
- **Student-Test Evaluation (Cell 09)**
  - Fit all 5 baseline models on the full 80 training trips.
  - Evaluate frozen baselines on the 10 student-test trips. Generate comparison tables and radar/bar charts.
- **Bayesian Hyperparameter Optimization (Cell 10)**
  - Pick the most promising model (e.g., HistGradientBoosting / Random Forest).
  - Use Optuna to optimize learning rate, tree depth, leaf count, and regularizations on a training `GroupKFold` objective.
  - Save trial logs and report Tuned vs. Untuned performance improvements.

**Maps to code:** `src/models/train.py`, `src/models/bayesian_opt.py`, `src/models/evaluate.py`, `artifacts/results/model_comparison.csv`, `artifacts/results/bayesian_optimisation.json`.

---

## Member 3 — HMM & Viterbi Route Reconstruction & Artifact Packaging

**Primary Environment:** Kaggle Notebook
**Notebook Cells Owned:** 11 — Decoder, 12 — Export
**Workflow Stages:** Stage 8 (HMM + Viterbi), Stage 9 (Model Artifacts & Export)

**Detailed Responsibilities:**

- **Network Topology & Graph Building**
  - Build directed graphs from `.nodes` and `.arcs` using NetworkX to compute shortest path distances and detect graph disconnections.
- **HMM Probabilistic Formulation (Cell 11)**
  - Emission Probability: combine geometric distance error with the ML model's predicted candidate probabilities:
    `log P(O_t | S_t) ∝ -1/2 (d / σ_z)^2 + log(P_ML)`
  - Transition Probability: penalize differences between shortest road path length and Euclidean GPS step displacement:
    `log P(S_t+1 | S_t) ∝ -|d_road - d_euclid| / β`
- **Viterbi Decoding & Route Reconstruction (Cell 11)**
  - Implement the dynamic programming Viterbi recurrence to reconstruct a topologically connected sequence of arcs.
  - Handle edge cases: missing candidates, invalid transitions, and large GPS time gaps (> 2s).
- **Decoder Benchmarking & Comparison**
  - Compare Classifier-Only vs. Classifier + HMM on the 10 student-test trips.
  - Report Route Continuity % (consecutive valid transitions) and sequence accuracy gains.
- **Artifact Packaging (Cell 12)**
  - Bundle the trained tuned model, feature columns list, scalers, `metadata.json`, and decoder settings into `pathsnap_artifacts.zip`.
  - Verify that artifacts can be cleanly unpickled/loaded in an external environment without errors.

**Maps to code:** `src/reconstruction/graph.py`, `src/reconstruction/viterbi.py`, `src/pipeline.py`, `src/models/sequence_eval.py`, `src/models/export.py`, `tests/test_candidate.py`, `tests/test_viterbi.py`.

---

## Member 4 — Two-Laptop Distributed System (FastAPI + Streamlit) & Technical Report Lead

**Primary Environment:** Local VS Code / Git / 2 Laptops
**Code Owned:** `src/api/` (Laptop 1), `ui/` (Laptop 2), `reports/` (Technical Report)
**Workflow Stages:** Stage 10 (FastAPI), Stage 11 (User Interface & Two-Laptop Integration), Stage 12 (Submission)

**Detailed Responsibilities:**

- **Laptop 1 Backend Engine (`src/api/main.py`)**
  - Build FastAPI endpoints (`GET /trips`, `POST /predict/{trip_id}`).
  - Load `pathsnap_artifacts.zip` to run the complete prediction pipeline (Preprocessing → Candidate scoring → HMM decoding).
  - Strict Isolation Rule: ground-truth `.route` files stay exclusively on Laptop 1. The API calculates accuracy internally and logs live inference requests on terminal.
- **Laptop 2 Evaluator UI (`ui/app.py`)**
  - Build the Streamlit interface displaying the 10 reserved evaluation trips.
  - Send the selected `trip_id` via HTTP request to Laptop 1 over the local Wi-Fi / LAN IP.
  - Render interactive Folium maps displaying:
    - Noisy GPS observations (red points)
    - Reconstructed predicted route (blue path)
    - Underlying road network (gray arcs)
    - Accuracy & Route Continuity scorecards
- **Two-Laptop Integration Testing**
  - Test network communication between Laptop 1 and Laptop 2.
  - Verify all 10 reserved evaluation trips execute successfully without hardcoded outputs.
- **Technical Report & Submission Lead**
  - Consolidate experimental results, EDA maps, the 5-model comparison table, Optuna curves, Classifier vs. HMM tables, and UI screenshots into `reports/technical_report.pdf`.
  - Ensure the GitHub repository has clean commit logs, setup instructions, and a comprehensive README.

**Maps to code:** `src/api/main.py`, `src/api/schemas.py`, `ui/app.py`, `src/visualization/map_plotter.py`, `README.md`, `notebooks/README.md`, `reports/`.

---
All members collaborated on design discussions, debugging, and the final technical report.
