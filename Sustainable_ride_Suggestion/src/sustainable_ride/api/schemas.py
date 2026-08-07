"""Request and response schemas for the HTTP API.

Validation lives in the schema rather than in the handler, so a bad request is
rejected before it reaches the models and the error message points at the
offending field.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from ..features.geo import NYC_BOUNDS


class Coordinate(BaseModel):
    lat: float = Field(..., description="Latitude in decimal degrees", examples=[40.7580])
    lon: float = Field(..., description="Longitude in decimal degrees", examples=[-73.9855])

    @field_validator("lat")
    @classmethod
    def _check_lat(cls, v: float) -> float:
        if not NYC_BOUNDS["lat_min"] <= v <= NYC_BOUNDS["lat_max"]:
            raise ValueError(
                f"latitude {v} is outside the NYC service area "
                f"[{NYC_BOUNDS['lat_min']}, {NYC_BOUNDS['lat_max']}]. The models "
                f"are trained on NYC trips only."
            )
        return v

    @field_validator("lon")
    @classmethod
    def _check_lon(cls, v: float) -> float:
        if not NYC_BOUNDS["lon_min"] <= v <= NYC_BOUNDS["lon_max"]:
            raise ValueError(
                f"longitude {v} is outside the NYC service area "
                f"[{NYC_BOUNDS['lon_min']}, {NYC_BOUNDS['lon_max']}]."
            )
        return v

    def as_tuple(self) -> tuple[float, float]:
        return (self.lat, self.lon)


class Weights(BaseModel):
    """User preference weights. Need not sum to 1 -- they are renormalised."""

    cost: float = Field(0.34, ge=0.0, le=1.0)
    time: float = Field(0.33, ge=0.0, le=1.0)
    co2: float = Field(0.33, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_least_one_positive(self) -> "Weights":
        if self.cost + self.time + self.co2 <= 0:
            raise ValueError("at least one weight must be greater than zero")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"cost": self.cost, "time": self.time, "co2": self.co2}


class RecommendRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    departure: datetime | None = Field(
        None, description="Departure time (ISO 8601). Defaults to now.")
    weights: Weights = Field(default_factory=Weights)
    passenger_count: int = Field(1, ge=1, le=6)
    rain_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description="0-1. At or above 0.6 the exposed modes are marked infeasible.")
    accessibility_required: bool = Field(
        False, description="Restrict to step-free modes.")
    include_geometry: bool = Field(
        True, description="Include route polylines in the response.")

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> "RecommendRequest":
        if (abs(self.origin.lat - self.destination.lat) < 1e-6
                and abs(self.origin.lon - self.destination.lon) < 1e-6):
            raise ValueError("origin and destination must be different locations")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "origin": {"lat": 40.7580, "lon": -73.9855},
                "destination": {"lat": 40.7061, "lon": -73.9969},
                "departure": "2024-01-15T17:30:00",
                "weights": {"cost": 0.2, "time": 0.3, "co2": 0.5},
                "passenger_count": 1,
                "rain_probability": 0.1,
            }]
        }
    }


class ModeOptionResponse(BaseModel):
    mode: str
    label: str
    duration_min: float
    vehicle_duration_min: float
    access_egress_min: float
    cost_usd: float
    co2_grams: float
    distance_km: float
    feasible: bool
    infeasible_reasons: list[str] = []
    warnings: list[str] = []
    score: float | None = None
    rank: int | None = None
    on_pareto_frontier: bool = False
    co2_saved_vs_taxi_g: float = 0.0
    cost_saved_vs_taxi_usd: float = 0.0
    time_delta_vs_taxi_min: float = 0.0
    duration_source: str = ""
    duration_confidence: str = "medium"
    cost_source: str = ""
    route_provider: str = ""
    route_is_estimate: bool = True
    fare_breakdown: dict = {}
    emission_detail: dict = {}
    geometry: list = []


class RecommendResponse(BaseModel):
    best_mode: str | None
    narrative: str
    frontier_modes: list[str]
    routing_provider: str
    weights: dict[str, float]
    departure: datetime
    options: list[ModeOptionResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    routing_provider: str
    version: str
