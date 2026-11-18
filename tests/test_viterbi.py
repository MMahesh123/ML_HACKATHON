"""Unit tests for the Viterbi route-reconstruction decoder."""

import pandas as pd

from src.reconstruction.graph import RoadNetworkGraph
from src.reconstruction.viterbi import viterbi_decode


def _chain_network():
    """0 -> 1 -> 2, a connected chain (arc 0, arc 1) plus a disconnected
    stub arc 2 (3 -> 4) that is not reachable from arc 0 at all."""
    nodes_df = pd.DataFrame({
        "longitude": [0.0, 0.0, 0.0, 10.0, 10.0],
        "latitude": [0.0, 0.001, 0.002, 0.0, 0.001],
        "node_id": [0, 1, 2, 3, 4],
    })
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1, 3],
        "to_node_id": [1, 2, 4],
        "arc_id": [0, 1, 2],
    })
    return RoadNetworkGraph(nodes_df, arcs_df)


def test_viterbi_prefers_connected_sequence_over_noisy_pointwise_best():
    road_graph = _chain_network()

    # Step 0: arc 0 is clearly best. Step 1: arc 2 is completely disconnected
    # from arc 0 (different component) and looks slightly better pointwise,
    # but the HMM should still favour the topologically connected arc 1.
    # Both arcs are ~110.6m long; placing each candidate at its arc midpoint
    # (t=0.5) means the connected hop 0 -> 1 covers exactly one arc length
    # (55.3m remaining on arc 0 + 55.3m already covered on arc 1 = 110.6m),
    # matching the observed GPS displacement almost exactly.
    trajectory_candidates = [
        [{"arc_id": 0, "distance": 1.0, "confidence": 0.9, "t": 0.5}],
        [
            {"arc_id": 2, "distance": 1.0, "confidence": 0.95, "t": 0.5},
            {"arc_id": 1, "distance": 2.0, "confidence": 0.85, "t": 0.5},
        ],
    ]
    gps_distances = [110.6]

    result = viterbi_decode(trajectory_candidates, gps_distances, road_graph)
    assert result[0] == 0
    assert result[1] == 1
    assert road_graph.are_arcs_connected(result[0], result[1])


def test_viterbi_handles_single_step():
    road_graph = _chain_network()
    result = viterbi_decode([[{"arc_id": 0, "distance": 0.5, "confidence": 1.0}]], [], road_graph)
    assert result == [0]


def test_viterbi_handles_empty_input():
    road_graph = _chain_network()
    assert viterbi_decode([], [], road_graph) == []
