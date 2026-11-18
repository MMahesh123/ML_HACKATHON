"""
Official evaluation metrics for PathSnap candidate classification
and observation-level map-matching.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.metrics import f1_score

def calculate_observation_top1_accuracy(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    target_col: str = "is_target",
    group_cols: List[str] = ["trip_id", "obs_idx"]
) -> float:
    """
    Computes observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, obs_idx), select the candidate with highest
    predicted score and verify whether it matches the aligned reference road arc.
    """
    if df.empty or prob_col not in df.columns:
        return 0.0

    # Sort so top candidate per observation is first
    sorted_df = df.sort_values(by=group_cols + [prob_col], ascending=[True, True, False])
    # Pick top-1 candidate per observation
    top1_df = sorted_df.groupby(group_cols, as_index=False).first()
    
    # Top-1 is correct if the selected candidate has is_target == 1
    return float(top1_df[target_col].mean())


def calculate_candidate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes candidate-row level classification metrics:
    Precision, Recall, F1, Accuracy, Log Loss.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    metrics = {
        "candidate_binary_precision": float(prec),
        "candidate_binary_recall": float(rec),
        "candidate_binary_f1": float(f1),
        "candidate_binary_accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if y_prob is not None:
        try:
            # Bound probabilities for numerical stability in log loss
            clipped_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
            metrics["log_loss"] = float(log_loss(y_true, clipped_prob))
        except Exception:
            metrics["log_loss"] = None

    return metrics


def calculate_route_continuity(predicted_arcs: List[int], road_graph: Optional[Any] = None) -> float:
    """
    Calculates route continuity: proportion of consecutive arc transitions
    that are valid adjacent segments in the road network graph.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    if road_graph is None:
        return 1.0

    valid_transitions = 0
    total_transitions = len(predicted_arcs) - 1

    for i in range(total_transitions):
        u_arc = predicted_arcs[i]
        v_arc = predicted_arcs[i + 1]
        if hasattr(road_graph, "are_arcs_connected") and road_graph.are_arcs_connected(u_arc, v_arc):
            valid_transitions += 1
        elif hasattr(road_graph, "shortest_path_distance"):
            if not np.isinf(road_graph.shortest_path_distance(u_arc, v_arc)):
                valid_transitions += 1

    return valid_transitions / total_transitions if total_transitions > 0 else 1.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    road_graph: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate candidate classification and ranking performance."""

    metrics = calculate_candidate_metrics(y_true, y_pred, y_prob)

    metrics["top_1_accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )

    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    if y_prob is not None:
        metrics["log_loss"] = float(
            log_loss(y_true, y_prob, labels=[0, 1])
        )

    if road_graph is not None:
        metrics["route_continuity"] = float(
            calculate_route_continuity(list(y_pred), road_graph)
        )

    return metrics
