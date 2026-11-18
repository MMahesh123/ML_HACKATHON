"""Viterbi sequential decoding algorithm for PathSnap route reconstruction."""

from typing import Dict, List, Tuple
import numpy as np
from src.config import BETA, DISCONNECTION_PENALTY, SIGMA_Z
from src.reconstruction.graph import RoadNetworkGraph


def emission_log_probability(distance_meters: float, model_confidence: float = 1.0) -> float:
    """
    Computes emission log probability based on GPS-to-segment distance and ML model probability.
    log P(O_t | S_t) ~ -0.5 * (d / sigma_z)^2 + log(P_ml)
    """
    geom_component = -0.5 * (distance_meters / SIGMA_Z) ** 2
    ml_component = np.log(max(model_confidence, 1e-6))
    return geom_component + ml_component


def transition_log_probability(
    from_cand: Dict,
    to_cand: Dict,
    great_circle_dist: float,
    road_graph: RoadNetworkGraph
) -> float:
    """
    Computes transition log probability between two candidate arcs.
    log P(S_t+1 | S_t) ~ - |d_road - d_euclid| / beta

    d_road is measured along the road from the candidate's own position on
    its arc (its projection fraction `t`) to the next candidate's position on
    its arc - not just the gap between the two arcs' endpoints, which would
    be ~0 for any directly-connected pair regardless of how far the vehicle
    actually travelled.
    """
    road_dist = road_graph.route_distance(
        from_cand["arc_id"], to_cand["arc_id"],
        from_t=from_cand.get("t", 1.0), to_t=to_cand.get("t", 0.0),
    )

    if np.isinf(road_dist):
        return np.log(DISCONNECTION_PENALTY)

    diff = abs(road_dist - great_circle_dist)
    return - (diff / BETA)


def viterbi_decode(
    trajectory_candidates: List[List[Dict]],
    gps_distances: List[float],
    road_graph: RoadNetworkGraph
) -> List[int]:
    """
    Applies the Viterbi algorithm to decode the maximum probability route sequence.

    Args:
        trajectory_candidates: List of candidate dicts for each time step.
            Each candidate dict has keys: 'arc_id', 'distance', 'confidence'.
        gps_distances: Great circle Euclidean distance between consecutive GPS points (length N-1).
        road_graph: RoadNetworkGraph instance.

    Returns:
        List of optimal arc_ids for the trajectory.
    """
    n_steps = len(trajectory_candidates)
    if n_steps == 0:
        return []

    # Initialize Viterbi matrices: viterbi_prob[t][candidate_idx]
    viterbi_prob = []
    backpointers = []

    # t = 0
    init_probs = []
    for cand in trajectory_candidates[0]:
        e_prob = emission_log_probability(cand["distance"], cand.get("confidence", 1.0))
        init_probs.append(e_prob)
    viterbi_prob.append(init_probs)
    backpointers.append([-1] * len(init_probs))

    # Recurrence for t = 1 to n_steps - 1
    for t in range(1, n_steps):
        current_cands = trajectory_candidates[t]
        prev_cands = trajectory_candidates[t - 1]
        step_gps_dist = gps_distances[t - 1]

        current_probs = []
        current_ptrs = []

        for curr_idx, curr_cand in enumerate(current_cands):
            e_prob = emission_log_probability(curr_cand["distance"], curr_cand.get("confidence", 1.0))

            best_prob = -float("inf")
            best_prev_idx = 0

            for prev_idx, prev_cand in enumerate(prev_cands):
                t_prob = transition_log_probability(
                    prev_cand,
                    curr_cand,
                    step_gps_dist,
                    road_graph
                )
                total_prob = viterbi_prob[t - 1][prev_idx] + t_prob + e_prob

                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev_idx = prev_idx

            current_probs.append(best_prob)
            current_ptrs.append(best_prev_idx)

        viterbi_prob.append(current_probs)
        backpointers.append(current_ptrs)

    # Termination: find best final state
    best_last_idx = int(np.argmax(viterbi_prob[-1]))

    # Backtracking
    best_path_indices = [best_last_idx]
    for t in range(n_steps - 1, 0, -1):
        prev_idx = backpointers[t][best_path_indices[-1]]
        best_path_indices.append(prev_idx)

    best_path_indices.reverse()

    # Map back to arc_id values
    optimal_route = [
        trajectory_candidates[t][idx]["arc_id"]
        for t, idx in enumerate(best_path_indices)
    ]

    return optimal_route
