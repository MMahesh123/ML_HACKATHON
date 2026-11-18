"""Preprocessing utilities: validation, cleaning, and route-to-track label alignment."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.geo import LocalProjector


def validate_track(track_df: pd.DataFrame) -> dict:
    """Returns a small data-quality report for a .track dataframe."""
    if track_df is None or len(track_df) == 0:
        return {"n_points": 0, "valid": False}

    ts = track_df["timestamp"].values
    dt = np.diff(ts)
    report = {
        "n_points": len(track_df),
        "valid": True,
        "n_duplicated_timestamps": int(np.sum(dt == 0)),
        "n_non_monotonic": int(np.sum(dt < 0)),
        "pct_1s_steps": float(np.mean(np.isclose(dt, 1.0))) if len(dt) else 1.0,
        "max_gap_s": float(np.max(dt)) if len(dt) else 0.0,
        "lon_range": (float(track_df["longitude"].min()), float(track_df["longitude"].max())),
        "lat_range": (float(track_df["latitude"].min()), float(track_df["latitude"].max())),
    }
    return report


def clean_track(track_df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate/non-monotonic timestamp rows and obviously invalid coordinates."""
    df = track_df.copy()
    df = df[(df["longitude"].abs() <= 180) & (df["latitude"].abs() <= 90)]
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    df = df.reset_index(drop=True)
    return df


def align_route_to_track(
    track_df: pd.DataFrame,
    route_arcs: Optional[np.ndarray],
    arcs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Aligns the chronological reference .route (one arc per route-step, not
    one per GPS fix) to the .track observations by nearest-position matching.

    Returns an array of length len(track_df) giving the reference arc_id for
    each GPS observation, or None if no route is available. Unmatched
    observations (should not normally happen) get -1.
    """
    if route_arcs is None or len(route_arcs) == 0:
        return None

    projector = LocalProjector.from_coords(nodes_df["longitude"].values, nodes_df["latitude"].values)
    node_xy = projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)

    arcs_indexed = arcs_df.set_index("arc_id")
    valid_route_arcs = [a for a in route_arcs if a in arcs_indexed.index]
    if not valid_route_arcs:
        return None

    route_from = arcs_indexed.loc[valid_route_arcs, "from_node_id"].values.astype(np.int64)
    route_to = arcs_indexed.loc[valid_route_arcs, "to_node_id"].values.astype(np.int64)
    route_mid = (node_xy[route_from] + node_xy[route_to]) / 2.0

    track_xy = projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)

    # For each GPS point, find the nearest point along the (ordered) route
    # polyline by nearest route-step midpoint, but constrained to be
    # monotonically non-decreasing in route order so the chronology of the
    # reference route is respected.
    labels = np.full(len(track_df), -1, dtype=np.int64)
    last_route_idx = 0
    n_route = len(valid_route_arcs)
    window = max(20, n_route // max(len(track_df), 1) + 10)

    for i in range(len(track_df)):
        lo = last_route_idx
        hi = min(n_route, last_route_idx + window)
        if hi <= lo:
            hi = n_route
        segment = route_mid[lo:hi]
        if len(segment) == 0:
            break
        dists = np.linalg.norm(segment - track_xy[i], axis=1)
        best = int(np.argmin(dists)) + lo
        labels[i] = valid_route_arcs[best]
        last_route_idx = best

    return labels
