"""Evaluation metrics for observation-level predictions and sequence route reconstruction."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from src.reconstruction.graph import RoadNetworkGraph


def calculate_route_continuity(predicted_arcs: List[int], road_graph: RoadNetworkGraph) -> float:
    """
    Calculates route continuity: proportion of consecutive arc transitions
    that are valid adjacent segments in the road network graph.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = len(predicted_arcs) - 1

    for i in range(total_transitions):
        u_arc = predicted_arcs[i]
        v_arc = predicted_arcs[i + 1]
        if road_graph.are_arcs_connected(u_arc, v_arc):
            valid_transitions += 1

    return valid_transitions / total_transitions


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    road_graph: Optional[RoadNetworkGraph] = None,
) -> Dict[str, float]:
    """
    Evaluates predictions against reference ground truth.
    Returns:
        - top_1_accuracy
        - macro_f1
        - route_continuity (if road_graph provided)
        - log_loss (if probabilities provided)
    """
    metrics = {
        "top_1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if road_graph is not None:
        metrics["route_continuity"] = float(calculate_route_continuity(list(y_pred), road_graph))

    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = None

    return metrics
