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
    one per GPS fix) to the .track observations.

    Uses an order-preserving dynamic-programming alignment: it finds the
    assignment of every GPS observation to a route step that minimises total
    perpendicular distance subject to the route index being monotonically
    non-decreasing. A greedy nearest-step scan cannot do this - an early
    mistaken jump forward is irreversible, so the alignment runs ahead of the
    vehicle and strands later observations on arcs hundreds of metres away.

    Returns an array of length len(track_df) giving the reference arc_id for
    each GPS observation, or None if no route is available.
    """
    if route_arcs is None or len(route_arcs) == 0:
        return None

    projector = LocalProjector.from_coords(nodes_df["longitude"].values, nodes_df["latitude"].values)
    node_xy = projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)

    arcs_indexed = arcs_df.set_index("arc_id")
    valid_route_arcs = np.asarray([a for a in route_arcs if a in arcs_indexed.index])
    if len(valid_route_arcs) == 0:
        return None

    node_pos = {int(nid): i for i, nid in enumerate(nodes_df["node_id"].values)}
    route_from = arcs_indexed.loc[valid_route_arcs, "from_node_id"].values
    route_to = arcs_indexed.loc[valid_route_arcs, "to_node_id"].values
    seg_a = node_xy[[node_pos[int(n)] for n in route_from]]
    seg_b = node_xy[[node_pos[int(n)] for n in route_to]]

    track_xy = projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)
    n_track, n_route = len(track_xy), len(valid_route_arcs)

    ab = seg_b - seg_a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ab_len2_safe = np.where(ab_len2 > 0, ab_len2, 1.0)

    def step_costs(point: np.ndarray) -> np.ndarray:
        """Perpendicular distance from one GPS point to every route segment."""
        ap = point - seg_a
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / ab_len2_safe, 0.0, 1.0)
        closest = seg_a + t[:, None] * ab
        return np.linalg.norm(point - closest, axis=1)

    route_idx = np.arange(n_route)
    backptr = np.empty((n_track, n_route), dtype=np.int32)
    prev = step_costs(track_xy[0])
    backptr[0] = route_idx

    for i in range(1, n_track):
        running_min = np.minimum.accumulate(prev)
        # Index of the best predecessor at or before each route step. Step 0 is
        # always a new minimum, so 0 is a safe fill value for the accumulate.
        best_prev = np.maximum.accumulate(np.where(prev <= running_min, route_idx, 0))
        prev = step_costs(track_xy[i]) + running_min
        backptr[i] = best_prev

    labels = np.empty(n_track, dtype=np.int64)
    j = int(np.argmin(prev))
    for i in range(n_track - 1, -1, -1):
        labels[i] = valid_route_arcs[j]
        if i > 0:
            j = int(backptr[i, j])

    return labels
