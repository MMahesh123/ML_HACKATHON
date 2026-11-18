"""Pydantic schemas for the FastAPI backend service."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TripPredictionRequest(BaseModel):
    trip_id: str = Field(..., description="The ID of the trip to reconstruct, e.g., 'trip_01'")


class GPSPoint(BaseModel):
    longitude: float
    latitude: float
    timestamp: float


class RouteSegment(BaseModel):
    arc_id: int
    from_node: int
    to_node: int
    from_coord: List[float]  # [lat, lon]
    to_coord: List[float]    # [lat, lon]


class EvaluationMetrics(BaseModel):
    top_1_accuracy: float
    macro_f1: float
    route_continuity: float
    inference_time_ms: float
    candidate_miss_rate: Optional[float] = None


class TripPredictionResponse(BaseModel):
    trip_id: str
    status: str
    predicted_arcs: List[int]
    metrics: EvaluationMetrics
    gps_trajectory: List[GPSPoint]
    predicted_route_coords: List[List[float]]  # List of [lat, lon] for UI plotting
    ground_truth_route_coords: Optional[List[List[float]]] = None  # Coordinate polylines for UI map rendering
