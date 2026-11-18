"""Feature engineering for (GPS observation, candidate arc) pairs.

Turns raw candidate geometry into the feature vector the classifier scores.
Every feature is computed in the trip's local planar projection so distances
and bearings are true metric quantities, not degrees.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.geo import LocalProjector, bearing_deg, angular_diff_deg

FEATURE_COLUMNS = [
    "perp_distance_m",
    "proj_fraction",
    "heading_diff_deg",
    "candidate_rank",
    "arc_length_m",
    "speed_mps",
    "time_gap_s",
    "prev_arc_connected",
]


def _trajectory_headings(xy: np.ndarray) -> np.ndarray:
    """Per-step heading (degrees) of the GPS track between consecutive points.
    Length N, first entry repeats the first real heading (no movement before point 0).
    """
    if len(xy) < 2:
        return np.zeros(len(xy))
    deltas = np.diff(xy, axis=0)
    headings = bearing_deg(deltas[:, 0], deltas[:, 1])
    return np.concatenate([[headings[0]], headings])


def build_features_for_trip(
    track_df: pd.DataFrame,
    candidates_per_point: List[List[dict]],
    arcs_df: pd.DataFrame,
    projector: LocalProjector,
    prev_chosen_arc_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Builds one feature row per (observation index, candidate) pair.

    Returns a DataFrame with columns: obs_idx, arc_id, + FEATURE_COLUMNS.
    `prev_chosen_arc_lookup` (arc_id -> to_node) is only used when building
    inference-time connectivity features online; for training-time feature
    extraction we approximate "prev_arc_connected" from the previous
    observation's *nearest* candidate as a weak signal (the real sequential
    consistency is learned by the HMM/Viterbi stage, not the classifier).
    """
    n_obs = len(track_df)
    xy = projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)
    headings = _trajectory_headings(xy)
    timestamps = track_df["timestamp"].values.astype(np.float64)

    rows = []
    prev_top_to_node = None
    for i in range(n_obs):
        cands = candidates_per_point[i]
        if not cands:
            continue
        time_gap = float(timestamps[i] - timestamps[i - 1]) if i > 0 else 0.0
        dist_prev = float(np.linalg.norm(xy[i] - xy[i - 1])) if i > 0 else 0.0
        speed = dist_prev / time_gap if time_gap > 1e-6 else 0.0

        for rank, cand in enumerate(cands):
            arc_vec_heading = cand.get("arc_heading_deg", headings[i])
            heading_diff = angular_diff_deg(headings[i], arc_vec_heading)

            prev_connected = 0.0
            if prev_top_to_node is not None and cand.get("from_node") == prev_top_to_node:
                prev_connected = 1.0

            rows.append({
                "obs_idx": i,
                "arc_id": cand["arc_id"],
                "perp_distance_m": cand["distance"],
                "proj_fraction": cand.get("t", 0.5),
                "heading_diff_deg": heading_diff,
                "candidate_rank": rank,
                "arc_length_m": cand.get("arc_length_m", 0.0),
                "speed_mps": speed,
                "time_gap_s": time_gap,
                "prev_arc_connected": prev_connected,
            })
        prev_top_to_node = cands[0].get("to_node")

    return pd.DataFrame(rows)


def attach_arc_headings(candidates_per_point: List[List[dict]], projector: LocalProjector, node_xy: np.ndarray) -> None:
    """In-place: adds 'arc_heading_deg' and 'arc_length_m' to every candidate dict."""
    for cands in candidates_per_point:
        for cand in cands:
            u, v = cand["from_node"], cand["to_node"]
            vec = node_xy[v] - node_xy[u]
            cand["arc_heading_deg"] = float(bearing_deg(vec[0], vec[1]))
            cand["arc_length_m"] = float(np.linalg.norm(vec))
