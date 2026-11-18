"""Unit tests for dataset loading and formatting."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import TripDataLoader


def test_load_track(tmp_path: Path):
    track_file = tmp_path / "test.track"
    # Format: lon lat timestamp
    track_file.write_text("77.5946 12.9716 1000.0\n77.5948 12.9718 1001.0\n")

    df = TripDataLoader.load_track(track_file)
    assert len(df) == 2
    assert list(df.columns) == ["longitude", "latitude", "timestamp"]
    assert df["longitude"].iloc[0] == pytest.approx(77.5946)
    assert df["timestamp"].iloc[1] == pytest.approx(1001.0)


def test_load_nodes(tmp_path: Path):
    nodes_file = tmp_path / "test.nodes"
    # Format: lon lat
    nodes_file.write_text("77.5900 12.9700\n77.5910 12.9710\n")

    df = TripDataLoader.load_nodes(nodes_file)
    assert len(df) == 2
    assert "node_id" in df.columns
    assert df["node_id"].tolist() == [0, 1]


def test_load_arcs(tmp_path: Path):
    arcs_file = tmp_path / "test.arcs"
    # Format: from_node_id to_node_id
    arcs_file.write_text("0 1\n1 2\n")

    df = TripDataLoader.load_arcs(arcs_file)
    assert len(df) == 2
    assert "arc_id" in df.columns
    assert df["from_node_id"].tolist() == [0, 1]
    assert df["to_node_id"].tolist() == [1, 2]


def test_load_route(tmp_path: Path):
    route_file = tmp_path / "test.route"
    # Format: arc_id
    route_file.write_text("0\n1\n")

    route = TripDataLoader.load_route(route_file)
    assert route is not None
    assert list(route) == [0, 1]
