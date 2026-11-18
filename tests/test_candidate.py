"""Unit tests for vectorised candidate generation."""

import numpy as np
import pandas as pd

from src.features.candidate import CandidateGenerator


def _simple_network():
    # A tiny 3-node, 2-arc network along a straight line, ~111m apart (0.001 deg lat).
    nodes_df = pd.DataFrame({
        "longitude": [0.0, 0.0, 0.0],
        "latitude": [0.0, 0.001, 0.002],
        "node_id": [0, 1, 2],
    })
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1],
        "to_node_id": [1, 2],
        "arc_id": [0, 1],
    })
    return nodes_df, arcs_df


def test_candidate_generation_finds_nearby_arc():
    nodes_df, arcs_df = _simple_network()
    gen = CandidateGenerator(nodes_df, arcs_df, radius_meters=50.0)

    # A point essentially on top of the first segment's midpoint.
    cands = gen.get_candidates_for_point(gps_lon=0.0, gps_lat=0.0005, top_k=5)
    assert len(cands) > 0
    assert cands[0]["arc_id"] == 0
    assert cands[0]["distance"] < 5.0  # should be very close (meters)


def test_candidate_generation_batch_matches_single_point():
    nodes_df, arcs_df = _simple_network()
    gen = CandidateGenerator(nodes_df, arcs_df, radius_meters=50.0)

    lons = [0.0, 0.0]
    lats = [0.0005, 0.0015]
    batch_results = gen.get_candidates_batch(lons, lats, top_k=5)

    assert len(batch_results) == 2
    assert batch_results[0][0]["arc_id"] == 0
    assert batch_results[1][0]["arc_id"] == 1


def test_candidate_generation_empty_network_returns_empty():
    nodes_df = pd.DataFrame({"longitude": [], "latitude": [], "node_id": []})
    arcs_df = pd.DataFrame({"from_node_id": [], "to_node_id": [], "arc_id": []})
    gen = CandidateGenerator(nodes_df, arcs_df)
    assert gen.get_candidates_for_point(0.0, 0.0) == []
