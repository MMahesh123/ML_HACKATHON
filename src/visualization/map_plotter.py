"""Folium-based geographical visualizer for trajectories, road network, and reconstructed routes."""

from pathlib import Path
from typing import List, Optional
import folium
import pandas as pd


def create_folium_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_arcs: Optional[List[int]] = None,
    ground_truth_route: Optional[List[int]] = None,
    output_html_path: Optional[Path] = None,
) -> folium.Map:
    """
    Generates an interactive Folium map visualizing:
    1. Raw GPS observations (Blue markers)
    2. Road network segments (Gray lines)
    3. Ground truth reference route (Green polyline)
    4. Reconstructed predicted route (Red dashed polyline)
    """
    # Center map on mean GPS coordinate
    mean_lat = track_df["latitude"].mean()
    mean_lon = track_df["longitude"].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles="CartoDB positron")

    # Layer 1: Road Network Geometry (Subtle gray lines)
    road_layer = folium.FeatureGroup(name="Road Network Graph", show=False)
    for _, arc in arcs_df.iterrows():
        u = int(arc["from_node_id"])
        v = int(arc["to_node_id"])
        u_node = nodes_df.loc[nodes_df["node_id"] == u]
        v_node = nodes_df.loc[nodes_df["node_id"] == v]
        if not u_node.empty and not v_node.empty:
            coords = [
                [u_node["latitude"].values[0], u_node["longitude"].values[0]],
                [v_node["latitude"].values[0], v_node["longitude"].values[0]],
            ]
            folium.PolyLine(coords, color="#888888", weight=1.5, opacity=0.5).add_to(road_layer)
    road_layer.add_to(m)

    # Layer 2: Noisy GPS Observations (Blue dots)
    gps_layer = folium.FeatureGroup(name="Observed GPS Trajectory", show=True)
    for _, pt in track_df.iterrows():
        folium.CircleMarker(
            location=[pt["latitude"], pt["longitude"]],
            radius=3,
            color="#0066cc",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.7,
            popup=f"Time: {pt.get('timestamp', 'N/A')}",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Helper function to extract segment polylines from arc IDs
    def extract_coords_from_arcs(arc_ids: List[int]) -> List[List[float]]:
        coords = []
        for arc_id in arc_ids:
            arc_row = arcs_df.loc[arcs_df["arc_id"] == arc_id]
            if not arc_row.empty:
                u = int(arc_row["from_node_id"].values[0])
                v = int(arc_row["to_node_id"].values[0])
                u_node = nodes_df.loc[nodes_df["node_id"] == u]
                v_node = nodes_df.loc[nodes_df["node_id"] == v]
                if not u_node.empty and not v_node.empty:
                    coords.append([u_node["latitude"].values[0], u_node["longitude"].values[0]])
                    coords.append([v_node["latitude"].values[0], v_node["longitude"].values[0]])
        return coords

    # Layer 3: Ground Truth Route (Solid Green)
    if ground_truth_route is not None:
        gt_layer = folium.FeatureGroup(name="Reference Ground Truth (.route)", show=True)
        gt_coords = extract_coords_from_arcs(ground_truth_route)
        if gt_coords:
            folium.PolyLine(gt_coords, color="#00aa44", weight=5, opacity=0.8, tooltip="Ground Truth Route").add_to(gt_layer)
        gt_layer.add_to(m)

    # Layer 4: PathSnap Predicted Route (Dashed Red/Orange)
    if predicted_arcs is not None:
        pred_layer = folium.FeatureGroup(name="PathSnap Reconstructed Route", show=True)
        pred_coords = extract_coords_from_arcs(predicted_arcs)
        if pred_coords:
            folium.PolyLine(pred_coords, color="#e63946", weight=3, dash_array="6, 6", tooltip="Predicted Route").add_to(pred_layer)
        pred_layer.add_to(m)

    # Add Layer Control for toggling visibility
    folium.LayerControl().add_to(m)

    if output_html_path:
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html_path))

    return m
