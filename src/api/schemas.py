"""Pydantic schemas for the FastAPI backend service."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TripPredictionRequest(BaseModel):
    trip_id: str = Field(..., description="The ID of the trip to reconstruct, e.g., '00000001'")


class GPSPoint(BaseModel):
    longitude: float
    latitude: float
    timestamp: float


class EvaluationMetrics(BaseModel):
    """Server-side-computed metrics only. Never includes raw ground-truth arcs."""
    top_1_accuracy: float
    macro_f1: Optional[float] = None
    route_continuity: float
    inference_time_ms: float
    candidate_miss_rate: Optional[float] = None
    model_name: Optional[str] = None


class TripPredictionResponse(BaseModel):
    trip_id: str
    status: str
    predicted_arcs: List[int]
    metrics: EvaluationMetrics
    gps_trajectory: List[GPSPoint]
    predicted_route_coords: List[List[float]]  # [lat, lon] pairs for UI plotting
