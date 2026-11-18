"""FastAPI backend server for PathSnap (Laptop 1).

Ground-truth `.route` files for the evaluator-testing partition are loaded
and used ONLY inside this process, to compute the metrics returned to the
UI. The raw reference arc sequence is never serialized into an API response
- the client only ever receives the predicted route and summary metrics.
"""

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    EvaluationMetrics,
    GPSPoint,
    TripPredictionRequest,
    TripPredictionResponse,
)
from src.config import EVALUATOR_TEST_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from src.data.loader import TripDataLoader
from src.data.preprocessor import align_route_to_track, clean_track
from src.models.evaluate import calculate_route_continuity, evaluate_predictions
from src.models.export import load_exported_model
from src.pipeline import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction Backend (Laptop 1)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at process startup, never per-request.
LOADED_MODEL = None


@app.on_event("startup")
def load_model_once():
    global LOADED_MODEL
    LOADED_MODEL = load_exported_model()
    if LOADED_MODEL is None:
        logger.warning(
            "No exported model found under artifacts/models/. "
            "Run `python -m scripts.run_full_pipeline` first. "
            "Falling back to geometry-only candidate ranking."
        )
    else:
        logger.info(f"Loaded model '{LOADED_MODEL.name}' for inference.")


def _find_trip_dir(trip_id: str):
    for partition_dir in [EVALUATOR_TEST_DIR, STUDENT_TEST_DIR, TRAIN_DIR]:
        candidate = partition_dir / trip_id
        if (candidate / f"{trip_id}.track").exists():
            return partition_dir, candidate
    return None, None


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "PathSnap Model & Route Reconstruction API",
        "model_loaded": LOADED_MODEL is not None,
        "model_name": LOADED_MODEL.name if LOADED_MODEL else None,
    }


@app.get("/trips", response_model=List[str])
def list_evaluation_trips():
    """Returns the list of reserved evaluator-testing trip IDs (falls back to
    the student-testing partition during development, before the evaluator
    partition is released)."""
    search_dir = EVALUATOR_TEST_DIR if EVALUATOR_TEST_DIR.exists() and any(EVALUATOR_TEST_DIR.iterdir()) else STUDENT_TEST_DIR
    if not search_dir.exists():
        return []

    trip_ids = sorted(p.stem for p in search_dir.glob("*/*.track"))
    return trip_ids


@app.post("/predict", response_model=TripPredictionResponse)
def predict_trip(req: TripPredictionRequest):
    """
    Processes the selected trip through the complete prediction pipeline:
    1. Loads GPS observations (.track) and road network (.nodes, .arcs).
    2. Runs spatial candidate generation + trained-classifier scoring.
    3. Executes HMM / Viterbi sequential route decoding.
    4. Computes server-side evaluation metrics against the reference route,
       if one is available for this trip (never returned to the client).
    5. Returns only the predicted route and summary metrics.
    """
    start_time = time.perf_counter()
    trip_id = req.trip_id
    logger.info(f"Received inference request for trip: {trip_id}")

    partition_dir, trip_dir = _find_trip_dir(trip_id)
    if trip_dir is None:
        logger.warning(f"Trip '{trip_id}' not found on disk.")
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    try:
        result = run_inference(trip_dir, trip_id, loaded_model=LOADED_MODEL)
    except Exception as exc:
        logger.exception(f"Inference failed for trip {trip_id}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    track_df = result["track_df"]
    nodes_df = result["nodes_df"]
    arcs_df = result["arcs_df"]
    predicted_arcs: List[Optional[int]] = result["predicted_arcs_hmm"]
    road_graph = result["road_graph"]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Server-side-only evaluation against the reference route, if this trip has one.
    raw = TripDataLoader.load_trip(trip_dir, trip_id)
    reference_route = raw["route"]
    metrics_dict = {"top_1_accuracy": None, "macro_f1": None}
    if reference_route is not None:
        reference_arcs = align_route_to_track(track_df, reference_route, arcs_df, nodes_df)
        if reference_arcs is not None:
            labeled_mask = reference_arcs >= 0
            y_true = reference_arcs[labeled_mask]
            y_pred = [predicted_arcs[i] if predicted_arcs[i] is not None else -1 for i in range(len(predicted_arcs))]
            y_pred = [y_pred[i] for i in range(len(y_pred)) if labeled_mask[i]]
            metrics_dict = evaluate_predictions(y_true, y_pred)

    clean_sequence = [a for a in predicted_arcs if a is not None]
    route_continuity = calculate_route_continuity(clean_sequence, road_graph) if len(clean_sequence) > 1 else 1.0

    gps_trajectory = [
        GPSPoint(longitude=row["longitude"], latitude=row["latitude"], timestamp=row["timestamp"])
        for _, row in track_df.iterrows()
    ]

    arc_lookup = arcs_df.set_index("arc_id")[["from_node_id", "to_node_id"]]
    node_latlon = nodes_df.set_index("node_id")[["latitude", "longitude"]]
    predicted_coords: List[List[float]] = []
    for arc_id in clean_sequence:
        if arc_id not in arc_lookup.index:
            continue
        u, v = arc_lookup.loc[arc_id, ["from_node_id", "to_node_id"]]
        if u in node_latlon.index and v in node_latlon.index:
            predicted_coords.append([float(node_latlon.loc[u, "latitude"]), float(node_latlon.loc[u, "longitude"])])
            predicted_coords.append([float(node_latlon.loc[v, "latitude"]), float(node_latlon.loc[v, "longitude"])])

    logger.info(
        f"Trip {trip_id} inference completed in {elapsed_ms:.2f} ms | "
        f"Top-1: {metrics_dict.get('top_1_accuracy')} | Continuity: {route_continuity:.4f} | "
        f"Model: {LOADED_MODEL.name if LOADED_MODEL else 'geometry-fallback'}"
    )

    return TripPredictionResponse(
        trip_id=trip_id,
        status="success",
        predicted_arcs=[a for a in predicted_arcs if a is not None],
        metrics=EvaluationMetrics(
            top_1_accuracy=metrics_dict.get("top_1_accuracy") or 0.0,
            macro_f1=metrics_dict.get("macro_f1"),
            route_continuity=route_continuity,
            inference_time_ms=elapsed_ms,
            model_name=LOADED_MODEL.name if LOADED_MODEL else "geometry-fallback",
        ),
        gps_trajectory=gps_trajectory,
        predicted_route_coords=predicted_coords,
    )
