"""Folium-based geographical visualiser for trajectories, road network, and reconstructed routes.

Road networks in this dataset can have 80k+ arcs; drawing every one of them
would produce an unusable, multi-hundred-MB HTML file. We instead draw only
the arcs whose midpoint falls inside the GPS track's bounding box (expanded
by a small margin), capped at `max_road_segments` for browser performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import folium
import numpy as np
import pandas as pd


def _node_lookup(nodes_df: pd.DataFrame) -> np.ndarray:
    """Returns an (N, 2) array of [lat, lon] indexed by node_id (row index)."""
    arr = np.full((int(nodes_df["node_id"].max()) + 1, 2), np.nan)
    arr[nodes_df["node_id"].values] = nodes_df[["latitude", "longitude"]].values
    return arr


def create_folium_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_arcs: Optional[List[int]] = None,
    ground_truth_route: Optional[List[int]] = None,
    output_html_path: Optional[Path] = None,
    max_road_segments: int = 4000,
    bbox_margin_deg: float = 0.01,
) -> folium.Map:
    """
    Generates an interactive Folium map visualising:
    1. Raw GPS observations (blue markers)
    2. Nearby road network segments (gray lines, bounding-box clipped)
    3. Ground truth reference route (green polyline), where available
    4. Reconstructed predicted route (red dashed polyline), where available
    """
    mean_lat = track_df["latitude"].mean()
    mean_lon = track_df["longitude"].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles="OpenStreetMap")

    node_latlon = _node_lookup(nodes_df)

    lat_min, lat_max = track_df["latitude"].min() - bbox_margin_deg, track_df["latitude"].max() + bbox_margin_deg
    lon_min, lon_max = track_df["longitude"].min() - bbox_margin_deg, track_df["longitude"].max() + bbox_margin_deg

    from_idx = arcs_df["from_node_id"].values
    to_idx = arcs_df["to_node_id"].values
    valid = (from_idx < len(node_latlon)) & (to_idx < len(node_latlon))
    from_idx, to_idx = from_idx[valid], to_idx[valid]

    u_latlon = node_latlon[from_idx]
    v_latlon = node_latlon[to_idx]
    mid_lat = (u_latlon[:, 0] + v_latlon[:, 0]) / 2.0
    mid_lon = (u_latlon[:, 1] + v_latlon[:, 1]) / 2.0

    in_bbox = (mid_lat >= lat_min) & (mid_lat <= lat_max) & (mid_lon >= lon_min) & (mid_lon <= lon_max)
    keep = np.where(in_bbox)[0]
    if len(keep) > max_road_segments:
        keep = np.random.RandomState(42).choice(keep, size=max_road_segments, replace=False)

    road_layer = folium.FeatureGroup(name=f"Nearby Road Network ({len(keep)} segments)", show=True)
    for i in keep:
        folium.PolyLine(
            [[u_latlon[i, 0], u_latlon[i, 1]], [v_latlon[i, 0], v_latlon[i, 1]]],
            color="#888888", weight=1.5, opacity=0.5,
        ).add_to(road_layer)
    road_layer.add_to(m)

    gps_layer = folium.FeatureGroup(name="Observed GPS Trajectory", show=True)
    for _, pt in track_df.iterrows():
        folium.CircleMarker(
            location=[pt["latitude"], pt["longitude"]],
            radius=3, color="#0066cc", fill=True, fill_color="#3399ff", fill_opacity=0.7,
            popup=f"Time: {pt.get('timestamp', 'N/A')}",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    arc_lookup = arcs_df.set_index("arc_id")[["from_node_id", "to_node_id"]]

    def extract_coords_from_arcs(arc_ids: List[int]) -> List[List[float]]:
        coords = []
        for arc_id in arc_ids:
            if arc_id is None or arc_id not in arc_lookup.index:
                continue
            u, v = arc_lookup.loc[arc_id, ["from_node_id", "to_node_id"]]
            if u < len(node_latlon) and v < len(node_latlon):
                coords.append([float(node_latlon[u, 0]), float(node_latlon[u, 1])])
                coords.append([float(node_latlon[v, 0]), float(node_latlon[v, 1])])
        return coords

    def continuous_runs(coords: List[List[float]]) -> List[List[List[float]]]:
        """Splits [start, end] arc pairs into physically continuous runs, so a
        polyline never draws a straight fake road across a gap."""
        runs: List[List[List[float]]] = []
        for start, end in zip(coords[0::2], coords[1::2]):
            if runs and runs[-1][-2:] == [start, end]:
                continue
            if runs and runs[-1][-1] == start:
                runs[-1].append(end)
            else:
                runs.append([start, end])
        return runs

    if ground_truth_route is not None:
        gt_layer = folium.FeatureGroup(name="Reference Ground Truth (.route)", show=True)
        for run in continuous_runs(extract_coords_from_arcs(ground_truth_route)):
            folium.PolyLine(run, color="#00aa44", weight=5, opacity=0.8, tooltip="Ground Truth Route").add_to(gt_layer)
        gt_layer.add_to(m)

    if predicted_arcs is not None:
        pred_layer = folium.FeatureGroup(name="PathSnap Reconstructed Route", show=True)
        for run in continuous_runs(extract_coords_from_arcs(predicted_arcs)):
            folium.PolyLine(run, color="#e63946", weight=3, dash_array="6, 6", tooltip="Predicted Route").add_to(pred_layer)
        pred_layer.add_to(m)

    folium.LayerControl().add_to(m)

    if output_html_path:
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html_path))

    return m
