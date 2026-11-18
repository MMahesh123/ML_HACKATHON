"""
Generates 4 tailored packages and zip files for the PathSnap team:
- Member 1: Data Engineering, EDA, Preprocessing, Spatial Candidates & Route Alignment Gate
- Member 2: 5-Model Benchmarking (GroupKFold CV) & Optuna Bayesian Hyperparameter Optimization
- Member 3: HMM & Viterbi Sequence Reconstruction, Route Continuity & Artifact Packaging
- Member 4: Two-Laptop Distributed Deployment (FastAPI + Streamlit UI) & Technical Report Lead
"""

import os
import shutil
import zipfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = BASE_DIR / "member_packages_staging"
OUTPUT_DIR = BASE_DIR / "distributed_member_zips"

def make_notebook(cells):
    """Generates standard Jupyter Notebook .ipynb JSON string."""
    nb_cells = []
    for cell_type, src in cells:
        nb_cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in src.strip().split("\n")],
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
        })
    return json.dumps({
        "cells": nb_cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.12"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }, indent=2)

def setup_directories():
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def copy_sample_trip(dest_folder, trip_id="00000001", partition="train"):
    src_trip_dir = BASE_DIR / "data" / "raw" / partition / trip_id
    dest_trip_dir = dest_folder / "data" / "sample_trips" / trip_id
    dest_trip_dir.mkdir(parents=True, exist_ok=True)
    if src_trip_dir.exists():
        for f in src_trip_dir.iterdir():
            shutil.copy2(f, dest_trip_dir / f.name)

# ==============================================================================
# MEMBER 1 BUILDER
# ==============================================================================
def build_member_1():
    m1_dir = STAGING_DIR / "member1_data_and_features"
    m1_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Guide
    guide_content = """# PathSnap — Member 1: Data Engineering, EDA & Sequence Alignment Gate

## Role & Objectives
You own **Phase 1 through Phase 4** (Workflow Stages 1–4, Kaggle Playbook sections 00–07):
1. **Trip Inventory & Validation:** Inspect 80 train + 10 student-test trips. Verify row counts, timestamps, coordinate bounds, and graph connectivity.
2. **Geospatial Preprocessing:** Project coordinates into meters (using UTM / pyproj or geodesic calculations). Never compute Euclidean distance directly on angular degrees.
3. **Candidate Generation:** Use KD-Tree spatial indexing to identify top candidate road segments within 50 meters for each GPS observation.
4. **MANDATORY GATE — Route Alignment:** `.track` (e.g. 1,315 points) and `.route` (e.g. 525 arcs) do NOT have 1:1 row correspondence! Implement a monotonic sequence alignment to map each GPS observation to its reference route arc before creating supervised labels.
5. **Feature Extraction:** Build $(X, y, \\text{groups})$ where $y \\in \\{0, 1\\}$ and `groups = trip_id`.

## Required Candidate Features:
- `point_segment_distance_m`: Perpendicular / shortest distance from GPS point to candidate road arc.
- `heading_difference_deg`: Angular difference between trajectory velocity vector and road segment bearing.
- `gps_speed_mps`: Instantaneous vehicle speed between consecutive GPS points.
- `segment_length_m`: Physical length of the candidate arc in meters.
- `segment_bearing_deg`: Direction of the directed arc.
- `time_gap_s`: Elapsed time since previous GPS observation.
- `candidate_rank_distance`: Rank of the candidate among neighbors by distance.

## Key Handoff to Member 2:
Save `train_candidates.csv` and `student_test_candidates.csv` containing:
`[trip_id, obs_idx, arc_id, is_target, ...feature_cols...]`
"""
    (m1_dir / "MEMBER_1_GUIDE.md").write_text(guide_content, encoding="utf-8")
    
    # 2. Notebook
    m1_nb_cells = [
        ("markdown", "# PathSnap — Stage 1: Data Preprocessing, EDA & Candidate Generation\n**Owner:** Member 1\nCovers Kaggle Playbook Sections 00 to 07."),
        ("code", """# Cell 00: Imports & Reproducibility
import os, glob, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from geopy.distance import geodesic

SEED = 42
np.random.seed(SEED)
print("Environment initialized. Ready for PathSnap Data Processing.")"""),
        ("code", """# Cell 01: Build Trip Inventory
def read_trip_file(path, names):
    return pd.read_csv(path, sep=r"\\s+", header=None, names=names, engine="python")

# Discover all trips
data_root = Path("data/sample_trips")  # or /kaggle/input/...
inventory = []
for trip_dir in sorted(data_root.glob("*/")):
    tid = trip_dir.name
    paths = {ext: trip_dir / f"{tid}.{ext}" for ext in ["track", "nodes", "arcs", "route"]}
    if all(p.exists() for p in paths.values()):
        track = read_trip_file(paths["track"], ["lon", "lat", "time"])
        nodes = read_trip_file(paths["nodes"], ["lon", "lat"])
        arcs = read_trip_file(paths["arcs"], ["from_node", "to_node"])
        route = read_trip_file(paths["route"], ["arc_id"])
        inventory.append({
            "trip": tid,
            "gps_rows": len(track),
            "node_rows": len(nodes),
            "arc_rows": len(arcs),
            "route_rows": len(route)
        })

inventory_df = pd.DataFrame(inventory)
print("Trip Inventory:")
print(inventory_df)"""),
        ("code", """# Cell 02: Structural Validation & Gap Analysis
def validate_trip(track, nodes, arcs, route):
    report = {
        "track_nonfinite": int((~np.isfinite(track[["lon", "lat", "time"]])).any(axis=1).sum()),
        "timestamp_nonincreasing": int((track["time"].diff().dropna() <= 0).sum()),
        "timestamp_gap_gt_2s": int((track["time"].diff() > 2).sum()),
        "bad_arc_endpoints": int(((arcs["from_node"] < 0) | (arcs["from_node"] >= len(nodes)) |
                                 (arcs["to_node"] < 0) | (arcs["to_node"] >= len(nodes))).sum()),
        "bad_route_arc_ids": int(((route["arc_id"] < 0) | (route["arc_id"] >= len(arcs))).sum())
    }
    return report

print("Validation completed.")"""),
        ("code", """# Cell 03-05: KD-Tree Spatial Candidate Generation
class SpatialCandidateGenerator:
    def __init__(self, nodes_df, arcs_df, radius_meters=50.0):
        self.radius_meters = radius_meters
        # Precompute midpoint coordinates for each arc
        midpoints = []
        self.arc_list = []
        for idx, row in arcs_df.iterrows():
            u, v = int(row["from_node"]), int(row["to_node"])
            u_pt = nodes_df.iloc[u]
            v_pt = nodes_df.iloc[v]
            mid_lat = (u_pt["lat"] + v_pt["lat"]) / 2.0
            mid_lon = (u_pt["lon"] + v_pt["lon"]) / 2.0
            midpoints.append([mid_lat, mid_lon])
            self.arc_list.append(idx)
        self.tree = KDTree(np.array(midpoints))

    def query(self, lat, lon, top_k=10):
        # Approximate degrees for radius search
        deg_r = (self.radius_meters * 1.5) / 111000.0
        indices = self.tree.query_ball_point([lat, lon], r=deg_r)
        if not indices:
            _, indices = self.tree.query([lat, lon], k=min(3, len(self.arc_list)))
            indices = [indices] if np.isscalar(indices) else list(indices)
        return [self.arc_list[i] for i in indices[:top_k]]

print("Spatial candidate generator initialized.")"""),
        ("code", """# Cell 06: Mandatory Gate — Sequence Alignment Algorithm
def align_route_to_gps(track_df, route_arcs, nodes_df, arcs_df):
    \"\"\"
    Monotonic sequence alignment mapping each GPS row to its reference route arc.
    Preserves chronological progress along the ground truth path.
    \"\"\"
    # Map each GPS point to closest arc in reference route sequence monotonically
    aligned_arc_labels = []
    route_ptr = 0
    n_route = len(route_arcs)
    
    for _, gps_row in track_df.iterrows():
        lat, lon = gps_row["lat"], gps_row["lon"]
        # Search locally around route_ptr to enforce order-preservation
        search_window = range(route_ptr, min(route_ptr + 8, n_route))
        best_arc = route_arcs.iloc[route_ptr]["arc_id"]
        best_dist = float("inf")
        best_idx = route_ptr
        
        for idx in search_window:
            arc_id = route_arcs.iloc[idx]["arc_id"]
            u = int(arcs_df.iloc[arc_id]["from_node"])
            v = int(arcs_df.iloc[arc_id]["to_node"])
            u_pt = nodes_df.iloc[u]
            v_pt = nodes_df.iloc[v]
            mid = ((u_pt["lat"] + v_pt["lat"])/2.0, (u_pt["lon"] + v_pt["lon"])/2.0)
            d = geodesic((lat, lon), mid).meters
            if d < best_dist:
                best_dist = d
                best_arc = arc_id
                best_idx = idx
                
        route_ptr = best_idx
        aligned_arc_labels.append(best_arc)
        
    return np.array(aligned_arc_labels)

print("Alignment gate algorithm ready.")"""),
        ("code", """# Cell 07: Feature Extraction Matrix Builder
feature_cols = [
    "point_segment_distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "segment_length_m",
    "segment_bearing_deg",
    "time_gap_s",
    "candidate_rank_distance",
]

print("Ready to extract candidate features and assemble (X, y, groups).")""")
    ]
    
    (m1_dir / "notebooks").mkdir(exist_ok=True)
    (m1_dir / "notebooks" / "01_eda_and_feature_pipeline.ipynb").write_text(make_notebook(m1_nb_cells), encoding="utf-8")
    
    # Copy source files
    src_dir = m1_dir / "src"
    src_dir.mkdir(exist_ok=True)
    shutil.copytree(BASE_DIR / "src" / "data", src_dir / "data")
    shutil.copytree(BASE_DIR / "src" / "features", src_dir / "features")
    shutil.copy2(BASE_DIR / "src" / "config.py", src_dir / "config.py")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    
    copy_sample_trip(m1_dir, "00000001", "train")
    shutil.copy2(BASE_DIR / "requirements.txt", m1_dir / "requirements.txt")

# ==============================================================================
# MEMBER 2 BUILDER
# ==============================================================================
def build_member_2():
    m2_dir = STAGING_DIR / "member2_models_and_tuning"
    m2_dir.mkdir(parents=True, exist_ok=True)
    
    guide_content = """# PathSnap — Member 2: 5-Model Benchmarking & Optuna Bayesian Optimization

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
"""
    (m2_dir / "MEMBER_2_GUIDE.md").write_text(guide_content, encoding="utf-8")
    
    m2_nb_cells = [
        ("markdown", "# PathSnap — Stage 2: Five-Model Benchmarking & Bayesian Optimization\n**Owner:** Member 2\nCovers Kaggle Playbook Sections 08 to 10."),
        ("code", """# Cell 08: Imports, Pipeline Definitions & GroupKFold
import time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import optuna
import joblib

SEED = 42

models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))
    ]),
    "DecisionTree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=SEED))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=SEED))
    ]),
    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED))
    ])
}
print(f"Loaded {len(models)} model pipelines.")"""),
        ("code", """# Synthetic Feature Sample for Immediate Testing
# (Replace with train_candidates.csv from Member 1)
np.random.seed(SEED)
n_samples = 2000
feature_cols = [
    "point_segment_distance_m", "heading_difference_deg", "gps_speed_mps",
    "segment_length_m", "segment_bearing_deg", "time_gap_s", "candidate_rank_distance"
]
X_sample = pd.DataFrame(np.random.randn(n_samples, len(feature_cols)), columns=feature_cols)
# Realistic relationships
X_sample["point_segment_distance_m"] = np.random.exponential(scale=15.0, size=n_samples)
X_sample["candidate_rank_distance"] = np.random.randint(1, 10, size=n_samples)
y_sample = (X_sample["point_segment_distance_m"] < 10.0).astype(int)
groups_sample = np.repeat(np.arange(20), n_samples // 20)

print(f"Sample candidate dataset: {X_sample.shape}, Positive labels: {y_sample.sum()}")"""),
        ("code", """# Cell 08.2: Run 5-Fold GroupKFold Cross-Validation
cv = GroupKFold(n_splits=5)
rows = []

for name, pipe in models.items():
    fold_scores = []
    fold_times = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_sample, y_sample, groups_sample), start=1):
        t0 = time.perf_counter()
        pipe.fit(X_sample.iloc[tr_idx], y_sample.iloc[tr_idx])
        fit_time = time.perf_counter() - t0
        
        preds = pipe.predict(X_sample.iloc[va_idx])
        f1 = f1_score(y_sample.iloc[va_idx], preds, zero_division=0)
        fold_scores.append(f1)
        fold_times.append(fit_time)
        
    rows.append({
        "Model": name,
        "CV_F1_Mean": np.mean(fold_scores),
        "CV_F1_Std": np.std(fold_scores),
        "Fit_Seconds_Mean": np.mean(fold_times)
    })

cv_summary = pd.DataFrame(rows)
print(cv_summary.to_string(index=False))"""),
        ("code", """# Cell 10: Optuna Bayesian Hyperparameter Optimization
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10.0, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "random_state": SEED
    }
    clf = HistGradientBoostingClassifier(**params)
    scores = []
    for tr_idx, va_idx in cv.split(X_sample, y_sample, groups_sample):
        clf.fit(X_sample.iloc[tr_idx], y_sample.iloc[tr_idx])
        pred = clf.predict(X_sample.iloc[va_idx])
        scores.append(f1_score(y_sample.iloc[va_idx], pred, zero_division=0))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("Best Parameters:")
print(study.best_params)
print(f"Best CV F1: {study.best_value:.4f}")""")
    ]
    
    (m2_dir / "notebooks").mkdir(exist_ok=True)
    (m2_dir / "notebooks" / "02_model_benchmarking_and_optuna.ipynb").write_text(make_notebook(m2_nb_cells), encoding="utf-8")
    
    src_dir = m2_dir / "src"
    src_dir.mkdir(exist_ok=True)
    shutil.copytree(BASE_DIR / "src" / "models", src_dir / "models")
    shutil.copy2(BASE_DIR / "src" / "config.py", src_dir / "config.py")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(BASE_DIR / "requirements.txt", m2_dir / "requirements.txt")

# ==============================================================================
# MEMBER 3 BUILDER
# ==============================================================================
def build_member_3():
    m3_dir = STAGING_DIR / "member3_hmm_viterbi_reconstruction"
    m3_dir.mkdir(parents=True, exist_ok=True)
    
    guide_content = """# PathSnap — Member 3: HMM & Viterbi Route Reconstruction & Artifact Packaging

## Role & Objectives
You own **Phase 8 and Phase 9** (Workflow Stages 8–9, Kaggle Playbook sections 11–12):
1. **Road Network Graph:** Construct a directed graph from `.nodes` and `.arcs` using NetworkX. Precompute shortest road distances between adjacent candidate segments.
2. **Emission Log-Probability:**
   $$\\log P(O_t \\mid S_t) = -\\frac{1}{2}\\left(\\frac{d}{\\sigma_z}\\right)^2 + \\log(P_{\\text{ML}})$$
3. **Transition Log-Probability:**
   $$\\log P(S_{t+1} \\mid S_t) = -\\frac{|d_{\\text{road}} - d_{\\text{euclid}}|}{\\beta}$$
   Apply a disconnection penalty floor (e.g. $10^{-6}$) for unreachable transitions.
4. **Viterbi Dynamic Programming Recurrence:** Backtrack the optimal, topologically connected sequence of road segments across the trajectory.
5. **Classifier-Only vs. Classifier + HMM Comparison:**
   Evaluate on the 10 student-test trips and prove gains in **Route Continuity %** and sequence accuracy.
6. **Artifact Packaging:** Create `pathsnap_artifacts.zip` containing:
   - `candidate_ranker.joblib`
   - `metadata.json`
   - `feature_cols.json`
   - `hmm_config.json`
"""
    (m3_dir / "MEMBER_3_GUIDE.md").write_text(guide_content, encoding="utf-8")
    
    m3_nb_cells = [
        ("markdown", "# PathSnap — Stage 3: HMM and Viterbi Route Reconstruction\n**Owner:** Member 3\nCovers Kaggle Playbook Sections 11 to 12."),
        ("code", """# Cell 11: Graph Builder, Emission, Transition & Viterbi Decoding
import numpy as np
import pandas as pd
import networkx as nx
import joblib

SIGMA_Z = 4.07   # GPS measurement noise std (meters)
BETA = 3.0       # Transition scale parameter
DISCONNECTION_PENALTY = 1e-6

class RoadGraph:
    def __init__(self, nodes_df, arcs_df):
        self.graph = nx.DiGraph()
        for idx, row in arcs_df.iterrows():
            u, v = int(row["from_node"]), int(row["to_node"])
            u_pt = (nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"])
            v_pt = (nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"])
            length = np.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1]) * 111000.0
            self.graph.add_edge(u, v, arc_id=idx, weight=length)

    def shortest_path(self, u_arc, v_arc):
        try:
            return nx.shortest_path_length(self.graph, u_arc, v_arc, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

print("Road graph and HMM parameters initialized.")"""),
        ("code", """# Viterbi Recurrence Implementation
def viterbi_decode(candidates_per_step, gps_dists, road_graph):
    n_steps = len(candidates_per_step)
    if n_steps == 0:
        return []
        
    viterbi_prob = []
    backpointers = []
    
    # Init step t = 0
    t0_probs = []
    for cand in candidates_per_step[0]:
        e_prob = -0.5 * (cand["dist"] / SIGMA_Z)**2 + np.log(max(cand.get("prob", 1.0), 1e-6))
        t0_probs.append(e_prob)
    viterbi_prob.append(t0_probs)
    backpointers.append([-1] * len(t0_probs))
    
    # Recurrence
    for t in range(1, n_steps):
        curr_cands = candidates_per_step[t]
        prev_cands = candidates_per_step[t-1]
        step_gps_d = gps_dists[t-1]
        
        curr_probs = []
        curr_ptrs = []
        
        for curr_c in curr_cands:
            e_prob = -0.5 * (curr_c["dist"] / SIGMA_Z)**2 + np.log(max(curr_c.get("prob", 1.0), 1e-6))
            best_val = -float("inf")
            best_prev = 0
            
            for p_idx, prev_c in enumerate(prev_cands):
                d_road = road_graph.shortest_path(prev_c["arc_id"], curr_c["arc_id"])
                if np.isinf(d_road):
                    t_prob = np.log(DISCONNECTION_PENALTY)
                else:
                    t_prob = -abs(d_road - step_gps_d) / BETA
                    
                total_prob = viterbi_prob[t-1][p_idx] + t_prob + e_prob
                if total_prob > best_val:
                    best_val = total_prob
                    best_prev = p_idx
                    
            curr_probs.append(best_val)
            curr_ptrs.append(best_prev)
            
        viterbi_prob.append(curr_probs)
        backpointers.append(curr_ptrs)
        
    # Backtrack
    best_last = int(np.argmax(viterbi_prob[-1]))
    path = [candidates_per_step[-1][best_last]["arc_id"]]
    
    for t in range(n_steps - 1, 0, -1):
        best_last = backpointers[t][best_last]
        path.append(candidates_per_step[t-1][best_last]["arc_id"])
        
    return path[::-1]

print("Viterbi sequence decoder ready.")"""),
        ("code", """# Route Continuity Metric
def calculate_continuity(predicted_arcs, arcs_df):
    if len(predicted_arcs) <= 1:
        return 1.0
    valid = 0
    total = len(predicted_arcs) - 1
    for i in range(total):
        curr_to = arcs_df.iloc[predicted_arcs[i]]["to_node"]
        next_from = arcs_df.iloc[predicted_arcs[i+1]]["from_node"]
        if curr_to == next_from:
            valid += 1
    return valid / total

print("Continuity evaluator ready.")""")
    ]
    
    (m3_dir / "notebooks").mkdir(exist_ok=True)
    (m3_dir / "notebooks" / "03_hmm_viterbi_route_reconstruction.ipynb").write_text(make_notebook(m3_nb_cells), encoding="utf-8")
    
    src_dir = m3_dir / "src"
    src_dir.mkdir(exist_ok=True)
    shutil.copytree(BASE_DIR / "src" / "reconstruction", src_dir / "reconstruction")
    shutil.copytree(BASE_DIR / "src" / "models", src_dir / "models")
    shutil.copy2(BASE_DIR / "src" / "config.py", src_dir / "config.py")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    
    copy_sample_trip(m3_dir, "00000001", "train")
    shutil.copy2(BASE_DIR / "requirements.txt", m3_dir / "requirements.txt")

# ==============================================================================
# MEMBER 4 BUILDER
# ==============================================================================
def build_member_4():
    m4_dir = STAGING_DIR / "member4_distributed_deployment_and_report"
    m4_dir.mkdir(parents=True, exist_ok=True)
    
    guide_content = """# PathSnap — Member 4: Two-Laptop Distributed Deployment & Technical Report Lead

## Role & Objectives
You own **Phase 10 through Phase 12** (Workflow Stages 10–12, Kaggle Playbook sections 13–16):
1. **Laptop 1 Backend (FastAPI):**
   - Run `python run_laptop1_api.py` (hosts on port `8000`).
   - Receives `POST /predict` containing `{"trip_id": "00000001"}`.
   - Executes the complete inference pipeline: Preprocessing $\\to$ Candidate generation $\\to$ Model scoring $\\to$ HMM Viterbi decoding.
   - Calculates evaluation metrics using internal server-side reference data.
   - **Security:** Ground-truth `.route` files remain strictly server-side on Laptop 1!
2. **Laptop 2 Frontend (Streamlit UI):**
   - Run `python run_laptop2_ui.py` (starts Streamlit on port `8501`).
   - Displays the 10 reserved evaluation trips in a dropdown.
   - Communicates with Laptop 1 over local Wi-Fi / LAN IP (e.g. `http://192.168.1.50:8000`).
   - Renders interactive **Folium** maps (GPS observations, road network, and predicted route).
3. **Two-Laptop Integration Testing:**
   - Run `python test_connection.py` to verify LAN communication.
   - Test all 10 reserved trips live.
4. **Technical Report Lead:**
   - Coordinate the final consolidated PDF report using `TECHNICAL_REPORT_TEMPLATE.md`.
"""
    (m4_dir / "MEMBER_4_GUIDE.md").write_text(guide_content, encoding="utf-8")
    
    # Launcher scripts
    (m4_dir / "run_laptop1_api.py").write_text("""import uvicorn
from src.config import API_PORT

if __name__ == "__main__":
    print("=" * 60)
    print("Starting PathSnap Backend Server (Laptop 1)...")
    print(f"API listening on 0.0.0.0:{API_PORT}")
    print("Show terminal API logs to the evaluator during viva.")
    print("=" * 60)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=API_PORT, reload=True)
""", encoding="utf-8")

    (m4_dir / "run_laptop2_ui.py").write_text("""import os, sys

if __name__ == "__main__":
    print("=" * 60)
    print("Starting PathSnap Evaluator UI (Laptop 2)...")
    print("=" * 60)
    os.system("streamlit run ui/app.py")
""", encoding="utf-8")

    (m4_dir / "test_connection.py").write_text("""import requests, sys

def test_api(ip="127.0.0.1", port=8000):
    url = f"http://{ip}:{port}"
    print(f"Testing connection to Laptop 1 at {url}...")
    try:
        r = requests.get(f"{url}/health", timeout=3)
        print("Health Check Response:", r.json())
        trips = requests.get(f"{url}/trips", timeout=3).json()
        print(f"Available Evaluation Trips ({len(trips)}):", trips[:5])
        print("SUCCESS: Network communication verified!")
    except Exception as e:
        print("CONNECTION FAILED:", e)

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    test_api(ip)
""", encoding="utf-8")

    # Technical report template
    report_template = """# PathSnap: Technical Evaluation Report
**Course:** UE24CS352A Machine Learning | PES University  
**Team Members:** Member 1, Member 2, Member 3, Member 4  

---

## 1. Executive Summary & Problem Formulation
- Summary of distributed map-matching and sequence reconstruction.
- System capabilities and architectural overview.

## 2. Dataset Exploration & Structural Quality Findings
- Inventory table of 80 training trips + 10 student-testing trips.
- GPS sampling rate distribution, timestamp intervals, and gap analysis (> 2s).
- Visual trajectory plots and Folium map examples.

## 3. Spatial Preprocessing & The Alignment Gate
- Justification of metric projection / geodesic distance vs. raw degree coordinates.
- Spatial candidate selection (KD-Tree search radius, candidate count, miss rate).
- **Mandatory Alignment Gate:** Method used to align chronological reference route sequences to GPS observations.

## 4. Five-Model Benchmarking & Comparison
- Comparative table across all 5 models (Logistic Regression, k-NN, Decision Tree, Random Forest, HistGradientBoosting):
  - 5-Fold GroupKFold CV F1-Score & Accuracy
  - Student-testing performance (frozen)
  - Precision, Recall, Macro F1, Log Loss
  - Training and inference latency

## 5. Bayesian Hyperparameter Optimization
- Optuna search space definition and objective function.
- Convergence history, parameter importances, and Tuned vs. Untuned comparison table.

## 6. HMM & Viterbi Sequence Reconstruction
- Hidden state, emission, and transition probability modeling.
- **Classifier-Only vs. Classifier + HMM** comparative results:
  - Observation Top-1 Accuracy
  - Route Continuity % (consecutive connected segments)
  - Handling of GPS gaps and network disconnections

## 7. Distributed Architecture (Laptop 1 & Laptop 2)
- Network layout, endpoint specifications, and server-side ground truth isolation.
- UI screenshots displaying Folium maps and evaluation scorecards.

## 8. Reproducibility & GitHub Structure
- Clean repository layout, dependency management, and reproducible execution commands.
"""
    (m4_dir / "reports").mkdir(exist_ok=True)
    (m4_dir / "reports" / "TECHNICAL_REPORT_TEMPLATE.md").write_text(report_template, encoding="utf-8")
    
    # Copy full source tree for deployment
    src_dir = m4_dir / "src"
    src_dir.mkdir(exist_ok=True)
    for pkg in ["api", "data", "features", "models", "reconstruction", "visualization"]:
        shutil.copytree(BASE_DIR / "src" / pkg, src_dir / pkg)
    shutil.copy2(BASE_DIR / "src" / "config.py", src_dir / "config.py")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    
    # Copy UI
    shutil.copytree(BASE_DIR / "ui", m4_dir / "ui")
    copy_sample_trip(m4_dir, "00000000", "student_test")
    shutil.copy2(BASE_DIR / "requirements.txt", m4_dir / "requirements.txt")

# ==============================================================================
# ZIP ARCHIVER
# ==============================================================================
def make_zips():
    print("Compressing packages into .zip archives...")
    for member_folder in sorted(STAGING_DIR.iterdir()):
        if member_folder.is_dir():
            zip_name = OUTPUT_DIR / f"{member_folder.name}.zip"
            with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(member_folder):
                    for file in files:
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(member_folder)
                        zf.write(full_path, rel_path)
            size_mb = zip_name.stat().st_size / (1024 * 1024)
            print(f"Created {zip_name.name} ({size_mb:.2f} MB)")

def main():
    setup_directories()
    print("Building Member 1 package...")
    build_member_1()
    print("Building Member 2 package...")
    build_member_2()
    print("Building Member 3 package...")
    build_member_3()
    print("Building Member 4 package...")
    build_member_4()
    make_zips()
    print("All 4 member packages successfully generated!")

if __name__ == "__main__":
    main()
