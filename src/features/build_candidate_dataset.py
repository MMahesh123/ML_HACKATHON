"""
Builds candidate-ranking datasets from raw trip partitions for PathSnap.
Follows the official Playbook requirements:
- KDTree candidate generation (top K nearest arcs)
- Metric coordinate projections & geodesic distance approximation
- Order-preserving monotonic route-to-GPS alignment
- Candidate-level feature extraction
- Group-aware labels (trip_id, obs_idx, is_target)
"""

import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.config import PROCESSED_DATA_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from src.data.loader import TripDataLoader


def extract_trip_candidates(
    trip_data: dict,
    k_candidates: int = 10,
) -> pd.DataFrame:
    """
    Extracts candidate road segments and engineered features for a single trip.
    """
    trip_id = trip_data["trip_id"]
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    if track is None or nodes is None or arcs is None or len(track) == 0:
        return pd.DataFrame()

    nodes_arr = nodes[["latitude", "longitude"]].values
    from_ids = arcs["from_node_id"].values
    to_ids = arcs["to_node_id"].values

    u_pts = nodes_arr[from_ids]
    v_pts = nodes_arr[to_ids]
    midpoints = (u_pts + v_pts) / 2.0

    # Directed arc geometry
    dlat = v_pts[:, 0] - u_pts[:, 0]
    dlon = v_pts[:, 1] - u_pts[:, 1]
    arc_bearings = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360
    # Mean latitude scaling for longitude distance
    mean_lat = np.mean(midpoints[:, 0]) if len(midpoints) > 0 else 12.97
    lat_scale = 111000.0
    lon_scale = 111000.0 * np.cos(np.radians(mean_lat))
    arc_lengths = np.hypot(dlat * lat_scale, dlon * lon_scale)

    tree = KDTree(midpoints)

    # GPS trajectory dynamics
    track_lat = track["latitude"].values
    track_lon = track["longitude"].values
    track_time = track["timestamp"].values
    n_pts = len(track)

    dt = np.diff(track_time, prepend=track_time[0])
    dt[dt <= 0] = 1.0  # Handle zero/negative time anomalies

    d_lat_trk = np.diff(track_lat, prepend=track_lat[0])
    d_lon_trk = np.diff(track_lon, prepend=track_lon[0])
    trk_bearings = (np.degrees(np.arctan2(d_lon_trk, d_lat_trk)) + 360) % 360
    trk_step_m = np.hypot(d_lat_trk * lat_scale, d_lon_trk * lon_scale)
    trk_speeds = trk_step_m / dt

    # Spatial KDTree search
    track_pts = np.column_stack([track_lat, track_lon])
    dists_deg, cand_indices = tree.query(track_pts, k=min(k_candidates, len(midpoints)))

    # Observation-to-Route Monotonic Alignment Gate
    aligned_target_arcs = [None] * n_pts
    if route is not None and len(route) > 0:
        route_ptr = 0
        n_route = len(route)
        for t in range(n_pts):
            pt = track_pts[t]
            search_window = range(route_ptr, min(route_ptr + 10, n_route))
            best_idx = route_ptr
            best_d = float("inf")
            for r_idx in search_window:
                r_arc = route[r_idx]
                r_mid = midpoints[r_arc]
                d = np.hypot((pt[0] - r_mid[0]) * lat_scale, (pt[1] - r_mid[1]) * lon_scale)
                if d < best_d:
                    best_d = d
                    best_idx = r_idx
            route_ptr = best_idx
            aligned_target_arcs[t] = route[best_idx]

    # Assemble candidate rows
    rows = []
    for t in range(n_pts):
        pt_lat = track_lat[t]
        pt_lon = track_lon[t]
        pt_speed = trk_speeds[t]
        pt_heading = trk_bearings[t]
        pt_dt = dt[t]
        true_arc = aligned_target_arcs[t]

        step_cands = cand_indices[t] if cand_indices.ndim > 1 else [cand_indices[t]]

        for rank, arc_id in enumerate(step_cands, start=1):
            arc_mid = midpoints[arc_id]
            # Approximate distance in meters
            dist_m = np.hypot((pt_lat - arc_mid[0]) * lat_scale, (pt_lon - arc_mid[1]) * lon_scale)

            # Heading difference in degrees [0, 180]
            arc_bearing = arc_bearings[arc_id]
            heading_diff = abs(pt_heading - arc_bearing) % 360
            if heading_diff > 180:
                heading_diff = 360 - heading_diff

            is_target = 1 if (true_arc is not None and arc_id == true_arc) else 0

            rows.append({
                "trip_id": trip_id,
                "obs_idx": t,
                "candidate_arc_id": arc_id,
                "point_segment_distance_m": float(dist_m),
                "heading_difference_deg": float(heading_diff),
                "gps_speed_mps": float(pt_speed),
                "segment_length_m": float(arc_lengths[arc_id]),
                "segment_bearing_deg": float(arc_bearing),
                "time_gap_s": float(pt_dt),
                "candidate_rank_distance": int(rank),
                "is_target": int(is_target),
            })

    return pd.DataFrame(rows)


def build_candidate_dataset(partition: str = "train", max_trips: int = None) -> pd.DataFrame:
    """
    Builds the complete candidate feature table across trips for a partition.
    """
    loader = TripDataLoader()
    part_dir = TRAIN_DIR if partition == "train" else STUDENT_TEST_DIR
    trip_folders = sorted([d for d in part_dir.iterdir() if d.is_dir()])

    if max_trips:
        trip_folders = trip_folders[:max_trips]

    print(f"Extracting candidates for {len(trip_folders)} {partition} trips...")
    all_dfs = []
    t0 = time.perf_counter()

    for idx, folder in enumerate(trip_folders, start=1):
        trip_id = folder.name
        trip_data = loader.load_trip(folder, trip_id)
        df_trip = extract_trip_candidates(trip_data, k_candidates=8)
        if not df_trip.empty:
            all_dfs.append(df_trip)

        if idx % 10 == 0 or idx == len(trip_folders):
            print(f"  Processed {idx}/{len(trip_folders)} trips (elapsed: {time.perf_counter() - t0:.1f}s)")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DATA_DIR / f"{partition}_candidates.parquet"
    combined_df.to_parquet(out_file, index=False)
    print(f"Saved {partition} candidate dataset: {combined_df.shape} -> {out_file}")
    return combined_df


if __name__ == "__main__":
    build_candidate_dataset("train", max_trips=30)
    build_candidate_dataset("student_test")
