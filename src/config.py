"""Global project configuration and path management for PathSnap."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Partitions
TRAIN_DIR = RAW_DATA_DIR / "train"
STUDENT_TEST_DIR = RAW_DATA_DIR / "student_test"
EVALUATOR_TEST_DIR = RAW_DATA_DIR / "evaluator_test"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
MAPS_DIR = ARTIFACTS_DIR / "maps"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Ensure runtime directories exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, SCALERS_DIR, MAPS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Candidate Generation Parameters
CANDIDATE_SEARCH_RADIUS_METERS = 50.0  # Search radius around GPS point for candidate arcs
MAX_CANDIDATES_PER_OBSERVATION = 10    # Maximum candidate arcs to consider per time step

# HMM Parameters
SIGMA_Z = 4.07       # GPS measurement noise standard deviation (meters)
BETA = 3.0          # Transition probability scale parameter
DISCONNECTION_PENALTY = 1e-6  # Probability floor for disconnected road segments

# API & Deployment Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")
