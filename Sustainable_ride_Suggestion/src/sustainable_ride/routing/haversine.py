"""Analytic router: straight-line distance scaled by an empirical circuity factor.

Circuity is the ratio of road distance to straight-line distance. It is usually
quoted as a literature constant, but we can do better: the TLC records carry a
metered ``trip_distance``, which *is* the road distance, alongside the pickup
and dropoff zones from which we can compute the straight-line distance. Dividing
one by the other over hundreds of thousands of trips gives a circuity factor
fitted to this city, on this road network, rather than borrowed from a paper
about somewhere else.

``python -m sustainable_ride.cli prepare`` performs that fit (see
:func:`sustainable_ride.data.preprocess.fit_circuity`) and writes the result to
``data/reference/circuity.json``; this module picks it up automatically.
"""

from __future__ import annotations

import json
import logging

from ..config import load_config, resolve_path
from ..features.geo import haversine_km
from .base import Profile, Route

logger = logging.getLogger(__name__)

# Free-flow speeds by profile, km/h. Only used to produce a nominal duration --
# the trained ML models supersede this for every mode we have data for, so these
# matter mainly as a sanity floor and for the geometry-free cold path.
NOMINAL_SPEED_KMH: dict[str, float] = {
    "driving-car": 21.0,      # NYC average taxi speed is famously poor
    "cycling-regular": 15.5,
    "cycling-electric": 19.0,
}

_PROFILE_TO_MODE = {
    "driving-car": "taxi",
    "cycling-regular": "bike",
    "cycling-electric": "scooter",
}


def load_circuity() -> dict[str, float]:
    """Circuity factors by mode, preferring the empirical fit over defaults."""
    cfg = load_config()
    factors = dict(cfg.get_path("routing.circuity_defaults", {}) or {})

    fitted_path = resolve_path("reference", "circuity.json")
    if fitted_path.exists():
        try:
            fitted = json.loads(fitted_path.read_text(encoding="utf-8"))
            for mode, value in (fitted.get("factors") or {}).items():
                if isinstance(value, (int, float)) and 1.0 <= value <= 3.0:
                    factors[mode] = float(value)
            logger.debug("Using empirically fitted circuity factors: %s", factors)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read fitted circuity (%s); using defaults", exc)

    return factors


class HaversineRouter:
    """Always-available router requiring no network and no API key."""

    name = "haversine+circuity"

    def __init__(self, circuity: dict[str, float] | None = None) -> None:
        self.circuity = circuity if circuity is not None else load_circuity()

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        profile: Profile,
    ) -> Route:
        straight_km = float(haversine_km(origin[0], origin[1],
                                         destination[0], destination[1]))
        mode = _PROFILE_TO_MODE.get(profile, "taxi")
        factor = float(self.circuity.get(mode, 1.4))
        road_km = straight_km * factor

        speed = NOMINAL_SPEED_KMH.get(profile, 20.0)
        duration_min = (road_km / speed) * 60.0 if speed > 0 else 0.0

        return Route(
            distance_km=road_km,
            duration_min=duration_min,
            geometry=(origin, destination),
            provider=self.name,
            is_estimate=True,
        )
