"""Routing abstraction.

The recommender only ever needs three things from a router: how far the trip is
by road, roughly how long a generic vehicle would take, and (optionally) the
geometry to draw. Keeping that behind an interface means the analytic fallback
and the live OpenRouteService client are interchangeable, and the models above
never learn which one they are talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

# Mode -> the kind of vehicle the router should assume.
Profile = str


@dataclass(frozen=True)
class Route:
    """A path between two points for one mode."""

    distance_km: float
    duration_min: float
    geometry: Sequence[tuple[float, float]] = field(default_factory=tuple)
    """Ordered ``(lat, lon)`` pairs. Empty when the router is analytic."""

    provider: str = "unknown"
    """Which router produced this, surfaced in the API response so consumers
    can tell a real road route from a straight-line estimate."""

    is_estimate: bool = True
    """True when the distance is inferred rather than measured along roads."""

    def __post_init__(self) -> None:
        if self.distance_km < 0:
            raise ValueError("distance_km must be non-negative")
        if self.duration_min < 0:
            raise ValueError("duration_min must be non-negative")


@runtime_checkable
class Router(Protocol):
    """Anything that can turn two coordinates into a :class:`Route`."""

    name: str

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        profile: Profile,
    ) -> Route:
        ...


class RoutingError(RuntimeError):
    """Raised when a router cannot produce a route at all."""
