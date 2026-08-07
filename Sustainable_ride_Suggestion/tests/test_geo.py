"""Geographic primitives."""

from __future__ import annotations

import numpy as np
import pytest

from sustainable_ride.features.geo import (
    bearing_deg,
    haversine_km,
    manhattan_km,
    validate_coordinate,
    within_nyc,
)

TIMES_SQUARE = (40.7580, -73.9855)
WALL_STREET = (40.7061, -73.9969)
JFK = (40.6413, -73.7781)


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(*TIMES_SQUARE, *TIMES_SQUARE) == pytest.approx(0.0, abs=1e-9)

    def test_known_distance(self):
        # Times Square to Wall Street is about 5.8 km as the crow flies.
        d = haversine_km(*TIMES_SQUARE, *WALL_STREET)
        assert d == pytest.approx(5.85, abs=0.25)

    def test_longer_known_distance(self):
        # Times Square to JFK is roughly 21 km straight-line.
        d = haversine_km(*TIMES_SQUARE, *JFK)
        assert d == pytest.approx(21.0, abs=1.0)

    def test_symmetric(self):
        forward = haversine_km(*TIMES_SQUARE, *JFK)
        backward = haversine_km(*JFK, *TIMES_SQUARE)
        assert forward == pytest.approx(backward)

    def test_vectorised_matches_scalar(self):
        lats = np.array([40.7580, 40.7061])
        lons = np.array([-73.9855, -73.9969])
        vector = haversine_km(lats, lons, JFK[0], JFK[1])
        assert vector[0] == pytest.approx(haversine_km(*TIMES_SQUARE, *JFK))
        assert vector[1] == pytest.approx(haversine_km(*WALL_STREET, *JFK))

    def test_antipodal_does_not_overflow(self):
        """Floating point can push the haversine term just above 1.0."""
        d = haversine_km(0.0, 0.0, 0.0, 180.0)
        assert np.isfinite(d)
        assert d == pytest.approx(20015.0, abs=5.0)


class TestBearing:
    def test_due_north(self):
        assert bearing_deg(40.70, -74.00, 40.80, -74.00) == pytest.approx(0.0, abs=0.5)

    def test_due_east(self):
        assert bearing_deg(40.70, -74.00, 40.70, -73.90) == pytest.approx(90.0, abs=0.5)

    def test_due_south(self):
        assert bearing_deg(40.80, -74.00, 40.70, -74.00) == pytest.approx(180.0, abs=0.5)

    def test_in_range(self):
        rng = np.random.default_rng(0)
        bearings = bearing_deg(rng.uniform(40.5, 41.0, 200), rng.uniform(-74.2, -73.7, 200),
                               rng.uniform(40.5, 41.0, 200), rng.uniform(-74.2, -73.7, 200))
        assert np.all((bearings >= 0.0) & (bearings < 360.0))


class TestManhattan:
    def test_never_shorter_than_straight_line(self):
        """L1 distance must dominate the great-circle distance."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            a = (rng.uniform(40.5, 41.0), rng.uniform(-74.2, -73.7))
            b = (rng.uniform(40.5, 41.0), rng.uniform(-74.2, -73.7))
            assert manhattan_km(*a, *b) >= haversine_km(*a, *b) - 1e-9


class TestBounds:
    def test_nyc_points_inside(self):
        for point in (TIMES_SQUARE, WALL_STREET, JFK):
            assert bool(within_nyc(*point))

    @pytest.mark.parametrize("point", [
        (51.5074, -0.1278),      # London
        (34.0522, -118.2437),    # Los Angeles
        (0.0, 0.0),              # Null Island
    ])
    def test_far_points_outside(self, point):
        assert not bool(within_nyc(*point))

    def test_validate_accepts_nyc(self):
        validate_coordinate(*TIMES_SQUARE)

    def test_validate_rejects_outside_nyc(self):
        with pytest.raises(ValueError, match="outside the NYC service area"):
            validate_coordinate(51.5074, -0.1278, "origin")

    def test_validate_rejects_impossible_latitude(self):
        with pytest.raises(ValueError, match=r"outside \[-90, 90\]"):
            validate_coordinate(120.0, -73.98)

    def test_validate_rejects_none(self):
        with pytest.raises(ValueError, match="required"):
            validate_coordinate(None, -73.98)
