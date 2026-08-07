"""FastAPI service exposing the recommender.

The models and the router are loaded once at startup rather than per request.
Deserialising a 3.5 MB joblib bundle takes long enough that doing it inside the
request path would dominate the latency of everything else here.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..emissions import factor_provenance
from ..models.registry import ModelRegistry, ModelsNotTrainedError
from ..recommender import RideRecommender
from .schemas import HealthResponse, RecommendRequest, RecommendResponse

logger = logging.getLogger(__name__)

_state: dict = {"recommender": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the models at startup; degrade to a clear error if untrained."""
    try:
        _state["recommender"] = RideRecommender()
        logger.info("Recommender ready (routing via %s)",
                    _state["recommender"].router.name)
    except ModelsNotTrainedError as exc:
        # Start anyway so /health can explain what is wrong, rather than
        # crash-looping with the reason buried in a container log.
        _state["error"] = str(exc)
        logger.error("Startup without models: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Sustainable Ride Suggestion API",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Predicts travel time, cost and lifecycle CO2 for taxi, bike and "
        "e-scooter trips in New York City, and ranks them against a "
        "traveller's cost/time/carbon priorities.\n\n"
        "Duration and fare models are gradient-boosted regressors trained on "
        "NYC TLC yellow-cab records and Citi Bike system data. Emission "
        "factors are sourced and returned alongside the estimates via "
        "`/emissions/factors`."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # demo service; tighten before any real deployment
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_recommender() -> RideRecommender:
    if _state.get("recommender") is None:
        raise HTTPException(
            status_code=503,
            detail=_state.get("error")
            or "Models are not loaded. Run `python -m sustainable_ride.cli train`.",
        )
    return _state["recommender"]


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness and readiness."""
    recommender = _state.get("recommender")
    return HealthResponse(
        status="ok" if recommender else "degraded",
        models_loaded=recommender is not None,
        routing_provider=recommender.router.name if recommender else "none",
        version=__version__,
    )


@app.get("/models", tags=["meta"])
def models() -> dict:
    """Model cards: estimator, training data, and held-out test metrics."""
    _get_recommender()
    return ModelRegistry.instance().describe()


@app.get("/emissions/factors", tags=["meta"])
def emissions_factors() -> dict:
    """The emission factors used, with citations.

    Exposed deliberately: a carbon figure a caller cannot trace to a source is
    not one they should rely on.
    """
    return {
        "factors": factor_provenance(),
        "note": (
            "Figures are lifecycle CO2-equivalent per passenger-kilometre, "
            "including a deadhead multiplier for the distance vehicles travel "
            "without a passenger."
        ),
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Rank travel modes for one trip."""
    recommender = _get_recommender()
    try:
        result = recommender.recommend(
            origin=request.origin.as_tuple(),
            destination=request.destination.as_tuple(),
            departure=request.departure or datetime.now(),
            weights=request.weights.as_dict(),
            passenger_count=request.passenger_count,
            rain_probability=request.rain_probability,
            accessibility_required=request.accessibility_required,
            include_geometry=request.include_geometry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc

    return RecommendResponse(
        best_mode=result.best.mode if result.best else None,
        narrative=result.narrative,
        frontier_modes=result.frontier_modes,
        routing_provider=result.routing_provider,
        weights=result.weights,
        departure=result.departure,
        options=[o.to_dict() for o in result.options],
    )


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Entry point used by ``cli serve``."""
    import uvicorn
    uvicorn.run("sustainable_ride.api.main:app", host=host, port=port, reload=reload)
