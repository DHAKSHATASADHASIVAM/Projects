"""Routing layer: live road routing with a graceful analytic fallback."""

from __future__ import annotations

import logging

from ..config import get_ors_api_key, load_config
from .base import Profile, Route, Router, RoutingError
from .haversine import HaversineRouter, load_circuity
from .ors import ORSRouter

logger = logging.getLogger(__name__)

__all__ = [
    "Profile", "Route", "Router", "RoutingError",
    "HaversineRouter", "ORSRouter", "FallbackRouter",
    "get_router", "profile_for_mode", "load_circuity",
]


def profile_for_mode(mode: str) -> Profile:
    """Map a travel mode onto the routing profile that best represents it."""
    cfg = load_config()
    profiles = cfg.get_path("routing.ors.profiles", {}) or {}
    return profiles.get(mode, "driving-car")


class FallbackRouter:
    """Tries a primary router, falls back to a secondary one on any failure.

    This is what makes live routing safe to depend on. When the key is missing,
    the quota is spent or the network is down, callers still get an answer --
    flagged as an estimate, so nothing downstream misrepresents its precision.
    """

    def __init__(self, primary: Router, secondary: Router) -> None:
        self.primary = primary
        self.secondary = secondary
        self._primary_failed = False

    @property
    def name(self) -> str:
        active = self.secondary if self._primary_failed else self.primary
        return f"{active.name} (fallback active)" if self._primary_failed else active.name

    def route(self, origin, destination, profile: Profile) -> Route:
        if not self._primary_failed:
            try:
                return self.primary.route(origin, destination, profile)
            except RoutingError as exc:
                # Latch the failure: once the key is bad or the quota is spent,
                # retrying every request just adds latency to every request.
                logger.warning(
                    "Primary router %s failed (%s). Falling back to %s for the "
                    "remainder of this session.",
                    self.primary.name, exc, self.secondary.name,
                )
                self._primary_failed = True
        return self.secondary.route(origin, destination, profile)


def get_router(provider: str | None = None) -> Router:
    """Build the router named by ``routing.provider`` (or the argument).

    ``auto`` -- the default -- uses OpenRouteService when a key is available and
    the analytic router otherwise, always with fallback wired in.
    """
    cfg = load_config()
    provider = (provider or cfg.get_path("routing.provider", "auto")).lower()
    analytic = HaversineRouter()

    if provider == "haversine":
        return analytic

    if provider in ("ors", "auto"):
        if not get_ors_api_key():
            if provider == "ors":
                raise RoutingError(
                    "routing.provider is 'ors' but no ORS_API_KEY is set. "
                    "Add one to .env, or set routing.provider to 'auto'."
                )
            logger.info(
                "No ORS_API_KEY found; using analytic routing. Distances will "
                "be straight-line estimates scaled by a fitted circuity factor."
            )
            return analytic
        try:
            return FallbackRouter(ORSRouter(), analytic)
        except RoutingError as exc:
            logger.warning("Could not initialise OpenRouteService (%s)", exc)
            return analytic

    raise ValueError(f"Unknown routing provider: {provider!r}")
