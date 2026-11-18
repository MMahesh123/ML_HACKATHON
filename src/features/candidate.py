"""Candidate generation and feature extraction for PathSnap."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial import KDTree


class CandidateGenerator:
    """
    Identifies candidate road segments (.arcs) for each GPS observation point (.track)
    within a spatial search radius.
    """

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, radius_meters: float = 50.0):
        self.nodes_df = nodes_df.copy()
        self.arcs_df = arcs_df.copy()
        self.radius_meters = radius_meters
        self._prepare_arc_geometries()

    def _prepare_arc_geometries(self):
        """Precomputes midpoints and KDTree for fast spatial candidate search."""
        midpoints = []
        arc_metadata = []

        for _, row in self.arcs_df.iterrows():
            arc_id = int(row["arc_id"])
            u = int(row["from_node_id"])
            v = int(row["to_node_id"])

            u_node = self.nodes_df.loc[self.nodes_df["node_id"] == u]
            v_node = self.nodes_df.loc[self.nodes_df["node_id"] == v]

            if not u_node.empty and not v_node.empty:
                u_lat, u_lon = u_node["latitude"].values[0], u_node["longitude"].values[0]
                v_lat, v_lon = v_node["latitude"].values[0], v_node["longitude"].values[0]

                mid_lat = (u_lat + v_lat) / 2.0
                mid_lon = (u_lon + v_lon) / 2.0

                midpoints.append([mid_lat, mid_lon])
                arc_metadata.append({
                    "arc_id": arc_id,
                    "u_coords": (u_lat, u_lon),
                    "v_coords": (v_lat, v_lon),
                    "mid_coords": (mid_lat, mid_lon)
                })

        self.midpoints = np.array(midpoints)
        self.arc_metadata = arc_metadata
        if len(midpoints) > 0:
            # Simple angular KDTree for candidate narrowing
            self.kdtree = KDTree(self.midpoints)
        else:
            self.kdtree = None

    def get_candidates_for_point(self, gps_lat: float, gps_lon: float, top_k: int = 10) -> List[Dict]:
        """
        Returns the top-k nearest candidate road segments with perpendicular/geodesic distance.
        """
        if self.kdtree is None or len(self.arc_metadata) == 0:
            return []

        # Convert radius from meters to approximate degrees (~111,000 meters per degree)
        deg_radius = (self.radius_meters * 1.5) / 111000.0
        indices = self.kdtree.query_ball_point([gps_lat, gps_lon], r=deg_radius)

        if not indices:
            # Fallback to nearest 3 neighbors if no arcs in search radius
            _, nearest = self.kdtree.query([gps_lat, gps_lon], k=min(3, len(self.arc_metadata)))
            indices = [nearest] if np.isscalar(nearest) else list(nearest)

        candidates = []
        for idx in indices:
            meta = self.arc_metadata[idx]
            dist = geodesic((gps_lat, gps_lon), meta["mid_coords"]).meters
            candidates.append({
                "arc_id": meta["arc_id"],
                "distance": dist,
                "confidence": 1.0 / (1.0 + dist / 10.0),
            })

        # Sort by distance and return top_k
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:top_k]
