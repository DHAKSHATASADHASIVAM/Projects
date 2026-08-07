"""Routing layer, including the fallback behaviour that keeps the service up."""

from __future__ import annotations

import pytest

from sustainable_ride.features.geo import haversine_km
from sustainable_ride.routing import (
    FallbackRouter,
    HaversineRouter,
    Route,
    RoutingError,
    profile_for_mode,
)

TIMES_SQUARE = (40.7580, -73.9855)
WALL_STREET = (40.7061, -73.9969)


class TestRoute:
    def test_rejects_negative_distance(self):
        with pytest.raises(ValueError, match="non-negative"):
            Route(distance_km=-1.0, duration_min=10.0)

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative"):
            Route(distance_km=1.0, duration_min=-10.0)


class TestHaversineRouter:
    def test_road_distance_exceeds_straight_line(self):
        """Circuity must inflate the straight-line distance, never shrink it."""
        router = HaversineRouter()
        route = router.route(TIMES_SQUARE, WALL_STREET, "driving-car")
        straight = haversine_km(*TIMES_SQUARE, *WALL_STREET)
        assert route.distance_km > straight

    def test_flags_itself_as_an_estimate(self):
        route = HaversineRouter().route(TIMES_SQUARE, WALL_STREET, "driving-car")
        assert route.is_estimate is True

    def test_bike_circuity_below_taxi(self):
        """Bikes use paths cars cannot, so their networks are less circuitous."""
        router = HaversineRouter({"taxi": 1.42, "bike": 1.28, "scooter": 1.28})
        taxi = router.route(TIMES_SQUARE, WALL_STREET, "driving-car")
        bike = router.route(TIMES_SQUARE, WALL_STREET, "cycling-regular")
        assert bike.distance_km < taxi.distance_km

    def test_duration_is_positive(self):
        route = HaversineRouter().route(TIMES_SQUARE, WALL_STREET, "driving-car")
        assert route.duration_min > 0

    def test_identical_points_give_zero_distance(self):
        route = HaversineRouter().route(TIMES_SQUARE, TIMES_SQUARE, "driving-car")
        assert route.distance_km == pytest.approx(0.0)

    def test_unknown_profile_falls_back_to_a_default(self):
        route = HaversineRouter().route(TIMES_SQUARE, WALL_STREET, "hovercraft")
        assert route.distance_km > 0


class _AlwaysFails:
    name = "always-fails"

    def __init__(self):
        self.calls = 0

    def route(self, origin, destination, profile):
        self.calls += 1
        raise RoutingError("simulated outage")


class TestFallbackRouter:
    def test_uses_primary_when_it_works(self):
        primary = HaversineRouter({"taxi": 2.0})
        secondary = HaversineRouter({"taxi": 1.0})
        route = FallbackRouter(primary, secondary).route(
            TIMES_SQUARE, WALL_STREET, "driving-car")
        straight = haversine_km(*TIMES_SQUARE, *WALL_STREET)
        assert route.distance_km == pytest.approx(straight * 2.0)

    def test_falls_back_on_failure(self):
        """A dead primary must degrade to an estimate, not raise."""
        router = FallbackRouter(_AlwaysFails(), HaversineRouter())
        route = router.route(TIMES_SQUARE, WALL_STREET, "driving-car")
        assert route.distance_km > 0
        assert route.is_estimate

    def test_failure_is_latched(self):
        """After one failure the primary is not retried on every request."""
        failing = _AlwaysFails()
        router = FallbackRouter(failing, HaversineRouter())
        for _ in range(5):
            router.route(TIMES_SQUARE, WALL_STREET, "driving-car")
        assert failing.calls == 1


class TestProfileMapping:
    @pytest.mark.parametrize("mode,expected", [
        ("taxi", "driving-car"),
        ("bike", "cycling-regular"),
        ("scooter", "cycling-electric"),
    ])
    def test_modes_map_to_profiles(self, mode, expected):
        assert profile_for_mode(mode) == expected

    def test_unknown_mode_defaults_to_driving(self):
        assert profile_for_mode("teleporter") == "driving-car"
