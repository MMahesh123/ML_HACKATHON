"""Vectorised candidate road-segment generation for PathSnap.

For every GPS observation we need the set of nearby road arcs ("candidates")
that it might actually be travelling on. The original approach (iterrows +
per-point geodesic calls) does not scale to trips whose road network has tens
of thousands of nodes/arcs. This version:

1. Builds arc endpoint/midpoint arrays once per trip in local planar meters.
2. Builds a single cKDTree over arc midpoints.
3. Answers candidate queries for *all* GPS points of a trip in one batched
   query, then refines with the true perpendicular point-to-segment distance
   (not just the midpoint distance) using vectorised numpy.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.geo import LocalProjector, point_segment_projection


class CandidateGenerator:
    """Identifies candidate arcs for GPS observations within a search radius."""

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, radius_meters: float = 50.0):
        self.nodes_df = nodes_df
        self.arcs_df = arcs_df
        self.radius_meters = radius_meters

        all_lons = np.concatenate([nodes_df["longitude"].values])
        all_lats = np.concatenate([nodes_df["latitude"].values])
        self.projector = LocalProjector.from_coords(all_lons, all_lats)

        node_xy = self.projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)
        # node_id is assumed to equal the row index (0-based), as per the dataset spec.
        self.node_xy = node_xy

        from_idx = arcs_df["from_node_id"].values.astype(np.int64)
        to_idx = arcs_df["to_node_id"].values.astype(np.int64)
        valid = (from_idx < len(node_xy)) & (to_idx < len(node_xy)) & (from_idx >= 0) & (to_idx >= 0)
        self._valid_arc_ids = arcs_df["arc_id"].values[valid]
        self._a = node_xy[from_idx[valid]]
        self._b = node_xy[to_idx[valid]]
        self._mid = (self._a + self._b) / 2.0
        self._from_node = from_idx[valid]
        self._to_node = to_idx[valid]

        if len(self._mid) > 0:
            self.kdtree = cKDTree(self._mid)
        else:
            self.kdtree = None

    def get_candidates_batch(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        top_k: int = 10,
        search_multiplier: float = 2.5,
    ) -> List[List[dict]]:
        """Returns, for every (lon, lat) GPS point, a list of candidate dicts.

        Each candidate dict has: arc_id, distance (m, perpendicular to segment),
        t (projection fraction along the arc), from_node, to_node.
        """
        n = len(lons)
        if self.kdtree is None or n == 0:
            return [[] for _ in range(n)]

        pts_xy = self.projector.to_xy(np.asarray(lons), np.asarray(lats))

        query_radius = self.radius_meters * search_multiplier
        neighbor_lists = self.kdtree.query_ball_point(pts_xy, r=query_radius)

        results: List[List[dict]] = []
        for i, idxs in enumerate(neighbor_lists):
            if len(idxs) == 0:
                # Fallback: nearest few midpoints regardless of radius.
                k = min(top_k, len(self._mid))
                _, nn_idx = self.kdtree.query(pts_xy[i], k=k)
                idxs = [nn_idx] if np.isscalar(nn_idx) else list(nn_idx)

            idxs = np.asarray(idxs, dtype=np.int64)
            p_rep = np.repeat(pts_xy[i][None, :], len(idxs), axis=0)
            dist, t = point_segment_projection(p_rep, self._a[idxs], self._b[idxs])

            order = np.argsort(dist)[:top_k]
            cands = []
            for j in order:
                arc_idx = idxs[j]
                cands.append({
                    "arc_id": int(self._valid_arc_ids[arc_idx]),
                    "distance": float(dist[j]),
                    "t": float(t[j]),
                    "from_node": int(self._from_node[arc_idx]),
                    "to_node": int(self._to_node[arc_idx]),
                })
            results.append(cands)
        return results

    def get_candidates_for_point(self, gps_lon: float, gps_lat: float, top_k: int = 10) -> List[dict]:
        """Single-point convenience wrapper (kept for API/UI call sites)."""
        return self.get_candidates_batch([gps_lon], [gps_lat], top_k=top_k)[0]
