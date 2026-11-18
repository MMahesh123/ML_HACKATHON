"""Data loader for PathSnap dataset files (.track, .nodes, .arcs, .route)."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class TripDataLoader:
    """Loads and formats trip files according to the PathSnap raw data specifications."""

    @staticmethod
    def load_track(track_path: Path) -> pd.DataFrame:
        """
        Loads .track file containing GPS observations.
        Columns: longitude, latitude, timestamp (nominally 1Hz)
        """
        df = pd.read_csv(
            track_path,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"],
            dtype={"longitude": np.float64, "latitude": np.float64, "timestamp": np.float64},
        )
        return df

    @staticmethod
    def load_nodes(nodes_path: Path) -> pd.DataFrame:
        """
        Loads .nodes file containing geographical road network nodes.
        Columns: longitude, latitude
        Row index (0-based) represents node_id.
        """
        df = pd.read_csv(
            nodes_path,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"],
            dtype={"longitude": np.float64, "latitude": np.float64},
        )
        df["node_id"] = df.index
        return df

    @staticmethod
    def load_arcs(arcs_path: Path) -> pd.DataFrame:
        """
        Loads .arcs file containing connections between road network nodes.
        Columns: from_node_id, to_node_id
        Row index (0-based) represents arc_id.
        """
        df = pd.read_csv(
            arcs_path,
            sep=r"\s+",
            header=None,
            names=["from_node_id", "to_node_id"],
            dtype={"from_node_id": np.int64, "to_node_id": np.int64},
        )
        df["arc_id"] = df.index
        return df

    @staticmethod
    def load_route(route_path: Path) -> Optional[np.ndarray]:
        """
        Loads .route file containing chronological sequence of ground-truth arc_id values.
        """
        if not route_path.exists():
            return None
        df = pd.read_csv(
            route_path,
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype={"arc_id": np.int64},
        )
        return df["arc_id"].values

    @classmethod
    def load_trip(cls, trip_dir: Path, trip_id: str) -> Dict:
        """
        Loads all available data for a single trip.
        Looks for files: <trip_id>.track, <trip_id>.nodes, <trip_id>.arcs, <trip_id>.route
        """
        track_file = trip_dir / f"{trip_id}.track"
        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"
        route_file = trip_dir / f"{trip_id}.route"

        track_df = cls.load_track(track_file) if track_file.exists() else None
        nodes_df = cls.load_nodes(nodes_file) if nodes_file.exists() else None
        arcs_df = cls.load_arcs(arcs_file) if arcs_file.exists() else None
        route_arr = cls.load_route(route_file)

        return {
            "trip_id": trip_id,
            "track": track_df,
            "nodes": nodes_df,
            "arcs": arcs_df,
            "route": route_arr,
        }
