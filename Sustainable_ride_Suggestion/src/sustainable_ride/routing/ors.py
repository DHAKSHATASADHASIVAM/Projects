"""OpenRouteService client -- real road geometry and road distances.

Requires a free API key in ``ORS_API_KEY`` (see ``.env.example``). The public
tier is rate limited to roughly 40 requests/minute and 2,000/day, so responses
are memoised: a recommendation for one origin-destination pair asks for three
profiles, and users tend to re-run the same query while moving the weight
sliders around.

Every failure path here degrades rather than raises. A rate limit, an expired
key or an outage should downgrade the answer to an analytic estimate, not take
the service down.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..config import get_ors_api_key, load_config
from .base import Profile, Route, RoutingError

logger = logging.getLogger(__name__)


class ORSRouter:
    """Live routing against the OpenRouteService directions API."""

    name = "openrouteservice"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None) -> None:
        cfg = load_config()
        self.api_key = api_key or get_ors_api_key()
        if not self.api_key:
            raise RoutingError(
                "No OpenRouteService API key. Set ORS_API_KEY in your .env "
                "file, or use the haversine router."
            )
        self.base_url = cfg.get_path("routing.ors.base_url",
                                     "https://api.openrouteservice.org").rstrip("/")
        self.timeout = float(cfg.get_path("routing.ors.timeout_s", 15))
        self._session = session or requests.Session()
        self._session.headers.update({
            "Authorization": self.api_key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/geo+json",
        })
        self._cache: dict[tuple, Route] = {}
        self._cache_limit = int(cfg.get_path("routing.ors.cache_size", 4096))

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _cache_key(origin, destination, profile) -> tuple:
        # ~11 m of precision is plenty; it also collapses jittery map clicks
        # onto the same cache entry.
        return (round(origin[0], 4), round(origin[1], 4),
                round(destination[0], 4), round(destination[1], 4), profile)

    def _request(self, profile: Profile, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v2/directions/{profile}/geojson"
        response = self._session.post(url, json=body, timeout=self.timeout)

        if response.status_code == 429:
            raise RoutingError("OpenRouteService rate limit reached")
        if response.status_code in (401, 403):
            raise RoutingError("OpenRouteService rejected the API key")
        if response.status_code >= 400:
            raise RoutingError(
                f"OpenRouteService returned {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response.json()

    # -- public API --------------------------------------------------------

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        profile: Profile,
    ) -> Route:
        key = self._cache_key(origin, destination, profile)
        if key in self._cache:
            return self._cache[key]

        # ORS speaks GeoJSON, which orders coordinates lon-then-lat.
        body = {
            "coordinates": [[origin[1], origin[0]], [destination[1], destination[0]]],
            "units": "km",
            "geometry": True,
        }

        try:
            payload = self._request(profile, body)
        except requests.RequestException as exc:
            raise RoutingError(f"OpenRouteService request failed: {exc}") from exc

        features = payload.get("features") or []
        if not features:
            raise RoutingError("OpenRouteService returned no route")

        feature = features[0]
        summary = (feature.get("properties") or {}).get("summary") or {}
        distance_km = float(summary.get("distance", 0.0))
        duration_min = float(summary.get("duration", 0.0)) / 60.0

        if distance_km <= 0:
            raise RoutingError("OpenRouteService returned a zero-length route")

        coords = (feature.get("geometry") or {}).get("coordinates") or []
        geometry = tuple((float(lat), float(lon)) for lon, lat in coords)

        route = Route(
            distance_km=distance_km,
            duration_min=duration_min,
            geometry=geometry,
            provider=self.name,
            is_estimate=False,
        )

        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = route
        return route
