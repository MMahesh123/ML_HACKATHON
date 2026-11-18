"""End-to-end PathSnap pipeline: shared by offline evaluation and the live API.

run_inference() is the single source of truth for turning a trip's raw
(.track, .nodes, .arcs) files into a reconstructed arc sequence. The API
layer and the offline evaluation scripts both call this function, so the
numbers reported in the technical report are guaranteed to match what the
live demo actually returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import CANDIDATE_SEARCH_RADIUS_METERS, MAX_CANDIDATES_PER_OBSERVATION
from src.data.loader import TripDataLoader
from src.data.preprocessor import clean_track
from src.features.candidate import CandidateGenerator
from src.features.extractor import FEATURE_COLUMNS, attach_arc_headings, build_features_for_trip
from src.reconstruction.graph import RoadNetworkGraph
from src.reconstruction.viterbi import viterbi_decode


@dataclass
class LoadedModel:
    """A trained classifier + the scaler/columns it expects, loaded once at API startup."""

    model: object
    scaler: object
    feature_columns: List[str]
    name: str


def run_inference(
    trip_dir: Path,
    trip_id: str,
    loaded_model: Optional[LoadedModel] = None,
    radius_m: float = CANDIDATE_SEARCH_RADIUS_METERS,
    top_k: int = MAX_CANDIDATES_PER_OBSERVATION,
):
    """Runs the full map-matching pipeline for one trip.

    Returns a dict with: track_df, predicted_arcs (classifier-only, one per
    observation), predicted_arcs_hmm (after Viterbi decoding), candidate_miss_rate,
    road_graph, and per-step candidate lists (for debugging/visualisation).
    """
    trip_data = TripDataLoader.load_trip(trip_dir, trip_id)
    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]

    if track_df is None or nodes_df is None or arcs_df is None:
        raise ValueError(f"Incomplete trip data for {trip_id}")

    track_df = clean_track(track_df)

    cand_gen = CandidateGenerator(nodes_df, arcs_df, radius_meters=radius_m)
    candidates_per_point = cand_gen.get_candidates_batch(
        track_df["longitude"].values, track_df["latitude"].values, top_k=top_k
    )
    attach_arc_headings(candidates_per_point, cand_gen.projector, cand_gen.node_xy)

    road_graph = RoadNetworkGraph(nodes_df, arcs_df)

    feats = build_features_for_trip(track_df, candidates_per_point, arcs_df, cand_gen.projector)

    # Score every candidate with the trained classifier (if provided); otherwise
    # fall back to a pure-geometry confidence score so the pipeline still runs
    # end to end before a model has been trained/exported.
    if loaded_model is not None and len(feats) > 0:
        X = feats[loaded_model.feature_columns].values
        X_scaled = loaded_model.scaler.transform(X) if loaded_model.scaler is not None else X
        proba = loaded_model.model.predict_proba(X_scaled)
        # Binary "is-correct-candidate" classifier: probability of class 1.
        classes = list(loaded_model.model.classes_)
        pos_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
        feats = feats.copy()
        feats["confidence"] = proba[:, pos_idx]
    else:
        feats = feats.copy()
        feats["confidence"] = 1.0 / (1.0 + feats["perp_distance_m"] / 10.0) if len(feats) else []

    confidence_lookup = {}
    for row in feats.itertuples(index=False):
        confidence_lookup[(row.obs_idx, row.arc_id)] = row.confidence

    for i, cands in enumerate(candidates_per_point):
        for cand in cands:
            cand["confidence"] = float(confidence_lookup.get((i, cand["arc_id"]), cand.get("confidence", 0.5)))

    # Classifier-only prediction: top confidence candidate per observation.
    predicted_arcs_classifier = []
    for cands in candidates_per_point:
        if not cands:
            predicted_arcs_classifier.append(None)
            continue
        best = max(cands, key=lambda c: c["confidence"])
        predicted_arcs_classifier.append(best["arc_id"])

    # GPS displacement between consecutive points (meters), for the HMM transition term.
    xy = cand_gen.projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)
    gps_distances = np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist() if len(xy) > 1 else []

    predicted_arcs_hmm: List[Optional[int]] = []
    if any(len(c) > 0 for c in candidates_per_point):
        # Viterbi needs a non-empty candidate list at every step; drop empty
        # steps (large GPS gaps / no nearby road) and re-insert None there.
        non_empty_idx = [i for i, c in enumerate(candidates_per_point) if c]
        compact_cands = [candidates_per_point[i] for i in non_empty_idx]
        compact_gps_dist = []
        for k in range(1, len(non_empty_idx)):
            i0, i1 = non_empty_idx[k - 1], non_empty_idx[k]
            compact_gps_dist.append(float(np.linalg.norm(xy[i1] - xy[i0])))

        decoded = viterbi_decode(compact_cands, compact_gps_dist, road_graph) if compact_cands else []
        result_map = dict(zip(non_empty_idx, decoded))
        predicted_arcs_hmm = [result_map.get(i) for i in range(len(candidates_per_point))]
    else:
        predicted_arcs_hmm = [None] * len(candidates_per_point)

    return {
        "track_df": track_df,
        "nodes_df": nodes_df,
        "arcs_df": arcs_df,
        "candidates_per_point": candidates_per_point,
        "predicted_arcs_classifier": predicted_arcs_classifier,
        "predicted_arcs_hmm": predicted_arcs_hmm,
        "road_graph": road_graph,
        "cand_gen": cand_gen,
    }
