"""Road network graph representation, built for speed on large per-trip networks.

Some trips in this dataset have 80k+ nodes and 160k+ arcs. Building the graph
with pandas .iterrows() and answering every shortest-path query with an
unbounded Dijkstra search is too slow to run interactively. This version:

- Builds the graph with a single vectorised edge list (no iterrows).
- Bounds every shortest-path search with a distance cutoff, since the HMM
  only ever needs the road distance between candidates that are already
  close to two consecutive (and therefore close-together) GPS fixes. A
  bounded Dijkstra explores only nodes within the cutoff, so its cost no
  longer depends on the size of the whole network.
- Caches single-source searches per unique "from" node within a trip, since
  the Viterbi recurrence repeatedly asks for distances from the same handful
  of previous-step candidate arcs.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from src.geo import LocalProjector


class RoadNetworkGraph:
    """Directed road network graph built from .nodes and .arcs."""

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, search_cutoff_m: float = 400.0):
        self.nodes_df = nodes_df
        self.arcs_df = arcs_df
        self.search_cutoff_m = search_cutoff_m

        self.projector = LocalProjector.from_coords(nodes_df["longitude"].values, nodes_df["latitude"].values)
        self.node_xy = self.projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)

        from_nodes = arcs_df["from_node_id"].values.astype(np.int64)
        to_nodes = arcs_df["to_node_id"].values.astype(np.int64)
        arc_ids = arcs_df["arc_id"].values.astype(np.int64)
        n_nodes = len(self.node_xy)
        valid = (from_nodes >= 0) & (from_nodes < n_nodes) & (to_nodes >= 0) & (to_nodes < n_nodes)

        lengths = np.linalg.norm(self.node_xy[to_nodes[valid]] - self.node_xy[from_nodes[valid]], axis=1)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(n_nodes))
        edges = zip(
            from_nodes[valid].tolist(),
            to_nodes[valid].tolist(),
            ({"arc_id": int(a), "length": float(l)} for a, l in zip(arc_ids[valid], lengths)),
        )
        self.graph.add_edges_from(edges)

        # arc_id -> (from_node, to_node) lookup, vectorised build.
        self._arc_endpoints: Dict[int, tuple] = {
            int(a): (int(u), int(v)) for a, u, v in zip(arc_ids[valid], from_nodes[valid], to_nodes[valid])
        }
        self._arc_length: Dict[int, float] = {
            int(a): float(l) for a, l in zip(arc_ids[valid], lengths)
        }
        self._dijkstra_cache: Dict[int, dict] = {}

    def _single_source_bounded(self, source_node: int) -> dict:
        if source_node in self._dijkstra_cache:
            return self._dijkstra_cache[source_node]
        try:
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, source_node, cutoff=self.search_cutoff_m, weight="length"
            )
        except nx.NodeNotFound:
            lengths = {}
        self._dijkstra_cache[source_node] = lengths
        return lengths

    def shortest_path_distance(self, from_arc_id: int, to_arc_id: int) -> float:
        """Road-network distance from the end of from_arc to the start of to_arc,
        bounded by search_cutoff_m. Returns inf if beyond the cutoff / unreachable.
        """
        if from_arc_id == to_arc_id:
            return 0.0
        endpoints_from = self._arc_endpoints.get(int(from_arc_id))
        endpoints_to = self._arc_endpoints.get(int(to_arc_id))
        if endpoints_from is None or endpoints_to is None:
            return float("inf")

        u = endpoints_from[1]  # to_node of from_arc
        v = endpoints_to[0]    # from_node of to_arc
        if u == v:
            return 0.0

        lengths = self._single_source_bounded(u)
        return lengths.get(v, float("inf"))

    def route_distance(
        self,
        from_arc_id: int,
        to_arc_id: int,
        from_t: float = 1.0,
        to_t: float = 0.0,
    ) -> float:
        """Along-the-road distance from a point at fraction `from_t` along
        from_arc to a point at fraction `to_t` along to_arc.

        This is the quantity that should be compared against the GPS
        great-circle displacement between two consecutive observations - not
        merely the gap between the two arcs' endpoints, which is ~0 for any
        directly connected pair and would otherwise make every ordinary
        forward-driving step look like a large, improbable jump.
        """
        if from_arc_id == to_arc_id:
            length = self._arc_length.get(int(from_arc_id), 0.0)
            return abs(to_t - from_t) * length

        endpoints_from = self._arc_endpoints.get(int(from_arc_id))
        endpoints_to = self._arc_endpoints.get(int(to_arc_id))
        if endpoints_from is None or endpoints_to is None:
            return float("inf")

        remaining_on_from = (1.0 - from_t) * self._arc_length.get(int(from_arc_id), 0.0)
        so_far_on_to = to_t * self._arc_length.get(int(to_arc_id), 0.0)

        u = endpoints_from[1]
        v = endpoints_to[0]
        hop = 0.0 if u == v else self._single_source_bounded(u).get(v, float("inf"))
        if np.isinf(hop):
            return float("inf")
        return remaining_on_from + hop + so_far_on_to

    def are_arcs_connected(self, from_arc_id: int, to_arc_id: int) -> bool:
        """True if from_arc's end node is exactly to_arc's start node."""
        if from_arc_id == to_arc_id:
            return True
        endpoints_from = self._arc_endpoints.get(int(from_arc_id))
        endpoints_to = self._arc_endpoints.get(int(to_arc_id))
        if endpoints_from is None or endpoints_to is None:
            return False
        return endpoints_from[1] == endpoints_to[0]
