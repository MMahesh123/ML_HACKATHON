"""Sequence-level (per-trip) evaluation: reuses the exact same pipeline the
live API calls, so the numbers in the report match the live demo.

For every trip we compare:
  - the classifier-only prediction (argmax confidence candidate per step)
  - the classifier + HMM/Viterbi prediction
against the reference route aligned onto the track (preprocessor.align_route_to_track).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.data.loader import TripDataLoader
from src.data.preprocessor import align_route_to_track, clean_track
from src.models.evaluate import calculate_route_continuity
from src.pipeline import LoadedModel, run_inference


def evaluate_sequence_on_trips(
    partition_dir: Path,
    trip_ids: List[str],
    loaded_model: LoadedModel = None,
) -> dict:
    per_trip = []

    for trip_id in trip_ids:
        trip_dir = partition_dir / trip_id
        raw = TripDataLoader.load_trip(trip_dir, trip_id)
        track_df, nodes_df, arcs_df, route_arr = raw["track"], raw["nodes"], raw["arcs"], raw["route"]
        if track_df is None or route_arr is None:
            continue
        track_clean = clean_track(track_df)
        reference_arcs = align_route_to_track(track_clean, route_arr, arcs_df, nodes_df)
        if reference_arcs is None:
            continue

        result = run_inference(trip_dir, trip_id, loaded_model=loaded_model)
        pred_classifier = result["predicted_arcs_classifier"]
        pred_hmm = result["predicted_arcs_hmm"]
        road_graph = result["road_graph"]

        labeled_mask = reference_arcs >= 0
        n_labeled = int(labeled_mask.sum())
        if n_labeled == 0:
            continue

        clf_correct = sum(
            1 for i in range(len(reference_arcs))
            if labeled_mask[i] and pred_classifier[i] == reference_arcs[i]
        )
        hmm_correct = sum(
            1 for i in range(len(reference_arcs))
            if labeled_mask[i] and pred_hmm[i] == reference_arcs[i]
        )

        clf_seq = [a for a in pred_classifier if a is not None]
        hmm_seq = [a for a in pred_hmm if a is not None]

        per_trip.append({
            "trip_id": trip_id,
            "n_obs": n_labeled,
            "top1_acc_classifier": clf_correct / n_labeled,
            "top1_acc_hmm": hmm_correct / n_labeled,
            "route_continuity_classifier": calculate_route_continuity(clf_seq, road_graph) if len(clf_seq) > 1 else 1.0,
            "route_continuity_hmm": calculate_route_continuity(hmm_seq, road_graph) if len(hmm_seq) > 1 else 1.0,
        })

    if not per_trip:
        return {"per_trip": [], "mean_top1_acc_classifier": None, "mean_top1_acc_hmm": None}

    agg = {
        "per_trip": per_trip,
        "mean_top1_acc_classifier": float(np.mean([t["top1_acc_classifier"] for t in per_trip])),
        "mean_top1_acc_hmm": float(np.mean([t["top1_acc_hmm"] for t in per_trip])),
        "mean_route_continuity_classifier": float(np.mean([t["route_continuity_classifier"] for t in per_trip])),
        "mean_route_continuity_hmm": float(np.mean([t["route_continuity_hmm"] for t in per_trip])),
    }
    return agg
