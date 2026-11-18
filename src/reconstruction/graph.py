"""Road network graph representation using NetworkX."""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import geodesic


class RoadNetworkGraph:
    """Represents the directed road network graph built from .nodes and .arcs."""

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame):
        self.nodes_df = nodes_df.copy()
        self.arcs_df = arcs_df.copy()
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        # Add nodes with coordinates
        for _, row in self.nodes_df.iterrows():
            node_id = int(row["node_id"])
            self.graph.add_node(
                node_id,
                longitude=row["longitude"],
                latitude=row["latitude"]
            )

        # Add directed edges (arcs) with lengths
        for _, row in self.arcs_df.iterrows():
            arc_id = int(row["arc_id"])
            u = int(row["from_node_id"])
            v = int(row["to_node_id"])

            if u in self.graph and v in self.graph:
                u_coords = (self.graph.nodes[u]["latitude"], self.graph.nodes[u]["longitude"])
                v_coords = (self.graph.nodes[v]["latitude"], self.graph.nodes[v]["longitude"])
                length = geodesic(u_coords, v_coords).meters

                self.graph.add_edge(u, v, arc_id=arc_id, length=length)

    def shortest_path_distance(self, from_arc_id: int, to_arc_id: int) -> float:
        """
        Calculates shortest path distance on the road network between two arcs.
        Distance is measured from the to_node of from_arc to from_node of to_arc.
        """
        if from_arc_id == to_arc_id:
            return 0.0

        from_row = self.arcs_df.loc[self.arcs_df["arc_id"] == from_arc_id]
        to_row = self.arcs_df.loc[self.arcs_df["arc_id"] == to_arc_id]

        if from_row.empty or to_row.empty:
            return float("inf")

        u = int(from_row["to_node_id"].values[0])
        v = int(to_row["from_node_id"].values[0])

        if u == v:
            return 0.0

        try:
            dist = nx.shortest_path_length(self.graph, source=u, target=v, weight="length")
            return dist
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def are_arcs_connected(self, from_arc_id: int, to_arc_id: int) -> bool:
        """Checks if two consecutive arcs are directly topologically adjacent."""
        if from_arc_id == to_arc_id:
            return True
        from_row = self.arcs_df.loc[self.arcs_df["arc_id"] == from_arc_id]
        to_row = self.arcs_df.loc[self.arcs_df["arc_id"] == to_arc_id]
        if from_row.empty or to_row.empty:
            return False
        return int(from_row["to_node_id"].values[0]) == int(to_row["from_node_id"].values[0])
