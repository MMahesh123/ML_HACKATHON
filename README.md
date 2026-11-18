# PathSnap: Distributed Map-Matching & Route Reconstruction

> **PES University — Department of Computer Science and Engineering**
> **Course:** UE24CS352A: Machine Learning
> **Challenge:** PathSnap — End-to-End Distributed Map-Matching & Sequence Route Reconstruction

---

## 1. Project Overview

GPS tracking systems often suffer from measurement noise caused by urban canyons, multipath reflections, receiver clock errors, and low or irregular sampling frequencies. As a consequence, raw GPS points frequently appear to jump across buildings, drift into adjacent parallel streets, or exhibit unnatural velocity spikes.

**PathSnap** is an end-to-end, distributed machine learning system that reconstructs realistic, topologically valid vehicle trajectories by matching noisy GPS observations to an underlying road network graph.

### Key System Capabilities
- **Vectorised geospatial preprocessing & candidate generation:** projects WGS84 coordinates to a local planar frame per trip and uses a `cKDTree` over arc midpoints to find candidate road segments for every GPS point in one batched query — fast enough to handle the largest road network in this dataset (88k nodes / 165k arcs) in well under a second.
- **Candidate classification (6 models, 6 families):** Logistic Regression, distance-weighted k-NN, Decision Tree, Random Forest, LightGBM, and an MLP, benchmarked with group-aware (per-trip) cross-validation.
- **Bayesian hyperparameter optimisation:** Optuna search over the strongest model family, using group-aware CV on the 80 training trips only.
- **HMM + Viterbi decoding:** reconstructs a topologically consistent route using the classifier's candidate probabilities as emissions and bounded road-network shortest-path distances (measured from each candidate's own projected position on its arc, not just arc endpoints) as transitions.
- **Two-laptop distributed deployment:** Laptop 1 hosts the FastAPI model/HMM server; Laptop 2 runs the Streamlit evaluator UI with interactive Folium maps.

---

## 2. Repository Structure

```text
ML_HACKATHON/
├── data/
│   ├── raw/{train,student_test,evaluator_test}/<trip_id>/<trip_id>.{track,nodes,arcs,route}
│   ├── processed/                 # Cached candidate+feature tables (parquet) + quality logs
│   └── README.md
├── scripts/
│   ├── run_eda.py                 # Dataset inventory, quality log, sample Folium map
│   └── run_full_pipeline.py       # build_dataset -> train -> bayesian_opt -> export, end to end
├── src/
│   ├── config.py                  # Paths, candidate/HMM hyperparameters, API settings
│   ├── geo.py                     # Local planar projection + vectorised distance/bearing helpers
│   ├── pipeline.py                # Single source of truth: raw trip -> candidates -> model -> HMM
│   ├── data/
│   │   ├── loader.py              # Parses .track / .nodes / .arcs / .route
│   │   ├── preprocessor.py        # Cleaning, validation, route-to-track label alignment
│   │   └── build_dataset.py       # Builds the supervised candidate-classification dataset
│   ├── features/
│   │   ├── candidate.py           # Vectorised cKDTree candidate generation
│   │   └── extractor.py           # Feature engineering per (observation, candidate) pair
│   ├── models/
│   │   ├── train.py               # 6-model benchmark (CV + student-test + sequence-level)
│   │   ├── evaluate.py            # Accuracy/F1/log-loss/route-continuity metrics
│   │   ├── sequence_eval.py       # Per-trip end-to-end evaluation via src.pipeline
│   │   ├── bayesian_opt.py        # Optuna search + tuned-vs-untuned comparison
│   │   └── export.py              # Saves/loads the final model+scaler for the API
│   ├── reconstruction/
│   │   ├── graph.py               # Road network graph, bounded shortest-path queries
│   │   └── viterbi.py             # Emission/transition probabilities + Viterbi decoding
│   ├── visualization/
│   │   └── map_plotter.py         # Folium maps (bounding-box clipped for large networks)
│   └── api/
│       ├── schemas.py             # Pydantic request/response models (no ground truth fields)
│       └── main.py                # FastAPI endpoints (Laptop 1 server)
├── ui/
│   └── app.py                     # Streamlit evaluator interface (Laptop 2 client)
├── artifacts/
│   ├── models/                    # final_model.joblib, final_scaler.joblib, model_metadata.json
│   ├── maps/                      # Exported interactive Folium HTML maps
│   └── results/                   # model_comparison.csv, bayesian_optimisation.json, dataset_summary.json
├── reports/figures/                # Plots for the technical report PDF
├── tests/                         # test_data_loader.py, test_candidate.py, test_viterbi.py
├── requirements.txt
└── README.md
```

---

## 3. Dataset Specification

Each trip folder `data/raw/<partition>/<trip_id>/` holds four whitespace-separated, header-less files:

| File | Columns | Notes |
| :--- | :--- | :--- |
| `.track` | `longitude, latitude, timestamp` | Nominally 1 Hz; ~97% of steps exactly 1s apart |
| `.nodes` | `longitude, latitude` | 0-based row index = `node_id` |
| `.arcs`  | `from_node_id, to_node_id` | 0-based row index = `arc_id` |
| `.route` | `arc_id` | Chronological reference route (not infallible ground truth) |

Partitions: **80 training trips**, **10 student-testing trips** (both included under `data/raw/`), and **10 evaluator-testing trips** (released separately by organisers at `data/raw/evaluator_test/` just before final evaluation — never used for training, feature selection or hyperparameter choice).

---

## 4. Setup & Installation

```bash
git clone https://github.com/MMahesh123/ML_HACKATHON.git
cd ML_HACKATHON

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Place the raw dataset under `data/raw/train/`, `data/raw/student_test/`, and (once released) `data/raw/evaluator_test/`, each containing one subfolder per trip as described above.

---

## 5. Running the Offline Pipeline

Run everything in one command:
```bash
python -m scripts.run_full_pipeline
```
This builds the candidate dataset, benchmarks all 6 models, runs Bayesian optimisation on the strongest family, and exports `artifacts/models/final_model.joblib` + `final_scaler.joblib` for the API to load.

Or run each stage individually:
```bash
python -m scripts.run_eda                 # dataset inventory, quality log, sample Folium map
python -m src.data.build_dataset           # candidate generation + feature engineering -> data/processed/*.parquet
python -m src.models.train                 # 6-model benchmark -> artifacts/results/model_comparison.csv
python -m src.models.bayesian_opt          # Optuna search -> artifacts/results/bayesian_optimisation.json
```

Run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 6. Two-Laptop Evaluation Setup

Both laptops need this repository, the venv, and `pip install -r requirements.txt`. **Only Laptop 1** needs the raw dataset and the exported model artifacts (`artifacts/models/`) — Laptop 2 only ever talks to Laptop 1's API.

### Step 1 — Connect both laptops to the same network
Use the venue Wi-Fi, or create a mobile hotspot on one laptop and connect the other to it. A hotspot is more reliable on crowded event Wi-Fi.

### Step 2 — Laptop 1: find its IP address and start the API
```bash
# macOS
ipconfig getifaddr en0
# Windows (PowerShell)
ipconfig            # look for "IPv4 Address" under the active adapter
# Linux
hostname -I
```
Then, from the repo root with the venv active:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Keep this terminal visible during evaluation — the evaluator will be shown these request/response logs. Verify it's reachable from Laptop 1 itself: `http://localhost:8000/health`, and that it loaded a model (`"model_loaded": true` in the response) — if it's `false`, run `python -m scripts.run_full_pipeline` first.

### Step 3 — Laptop 2: point the UI at Laptop 1 and launch it
```bash
export API_URL="http://<laptop-1-ip>:8000"     # macOS/Linux
$env:API_URL="http://<laptop-1-ip>:8000"       # Windows PowerShell
streamlit run ui/app.py
```
(The API URL can also be typed directly into the sidebar text box in the UI.)

### Step 4 — Verify end to end
On Laptop 2's browser: confirm the trip dropdown populates from `/trips`, pick a trip, click **Reconstruct Route**, and confirm the map + metrics render. Check Laptop 1's terminal shows the matching `Received inference request` / `inference completed` log lines for that trip ID.

### Troubleshooting
- **Connection refused / timeout:** confirm both laptops show the same Wi-Fi/hotspot network; check firewall prompts on Laptop 1 allow incoming connections on port 8000 (the first `uvicorn` run usually triggers this prompt).
- **Trips dropdown empty:** `EVALUATOR_TEST_DIR`/`STUDENT_TEST_DIR` must exist and contain trip subfolders on Laptop 1.
- **`model_loaded: false`:** the API falls back to pure-geometry candidate ranking so the demo still runs, but run `python -m scripts.run_full_pipeline` on Laptop 1 and restart `uvicorn` before the real evaluation.

---

## 7. Evaluation Metrics Reported

- **Observation-level Top-1 Accuracy** — per GPS observation, does the highest-confidence candidate match the reference arc.
- **Macro F1-score**, **Log Loss** — candidate-classification quality (row-level, both classes).
- **Candidate Miss Rate** — fraction of observations where the reference arc wasn't in the candidate set at all (reported honestly in `data/processed/*_quality_log.json`).
- **Route Continuity** — fraction of consecutive predicted arcs that are directly connected in the road-network graph.
- **Training / Inference Time** — per model, in `artifacts/results/model_comparison.csv`.

All of these are computed by the exact same `src.pipeline.run_inference` function that the live API calls — the report's numbers and the live demo's numbers cannot drift apart.

---

## 8. Academic Integrity & Anti-Leakage Compliance

- Evaluator-testing data is never touched during feature engineering, training, or hyperparameter search.
- Evaluator ground-truth `.route` files stay strictly on Laptop 1; `src/api/schemas.py` has no field for them, and `src/api/main.py` never serialises them into a response.
- No hardcoded trip-id → prediction mapping anywhere in the codebase — every prediction is produced at request time by `src.pipeline.run_inference`.
- Live API request/response logs are shown on Laptop 1's console during evaluation.
