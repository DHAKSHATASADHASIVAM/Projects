"""Synthetic data generator.

This is the fallback path for anyone who clones the repo without downloading
420 MB of source data, so it needs to hold up: the schema must match the real
processed data exactly, and the structure the models rely on -- congestion,
distance-dependent speed -- must genuinely be present.
"""

from __future__ import annotations

import numpy as np
import pytest

from sustainable_ride.data.synthetic import generate_bike_trips, generate_taxi_trips
from sustainable_ride.features.geo import within_nyc


@pytest.fixture(scope="module")
def taxi():
    return generate_taxi_trips(n=6000, seed=42)


@pytest.fixture(scope="module")
def bike():
    return generate_bike_trips(n=6000, seed=42)


class TestSchema:
    def test_taxi_has_required_columns(self, taxi):
        required = {"pickup_datetime", "pu_lat", "pu_lon", "do_lat", "do_lon",
                    "PULocationID", "DOLocationID", "passenger_count",
                    "distance_km", "duration_min", "fare_amount"}
        assert required <= set(taxi.columns)

    def test_bike_has_required_columns(self, bike):
        required = {"started_at", "start_lat", "start_lng", "end_lat", "end_lng",
                    "rideable_type", "distance_km", "duration_min"}
        assert required <= set(bike.columns)

    def test_requested_row_count(self, taxi, bike):
        assert len(taxi) == 6000
        assert len(bike) == 6000

    def test_no_missing_values(self, taxi, bike):
        assert not taxi.isna().any().any()
        assert not bike.isna().any().any()


class TestPlausibility:
    def test_coordinates_land_in_nyc(self, taxi, bike):
        assert within_nyc(taxi.pu_lat, taxi.pu_lon).all()
        assert within_nyc(taxi.do_lat, taxi.do_lon).all()
        assert within_nyc(bike.start_lat, bike.start_lng).all()

    def test_positive_quantities(self, taxi, bike):
        assert (taxi.duration_min > 0).all()
        assert (taxi.distance_km > 0).all()
        assert (taxi.fare_amount > 0).all()
        assert (bike.duration_min > 0).all()

    def test_taxi_speeds_are_realistic(self, taxi):
        speed = taxi.distance_km / (taxi.duration_min / 60.0)
        assert 8.0 < speed.median() < 40.0

    def test_bike_speeds_are_realistic(self, bike):
        speed = bike.distance_km / (bike.duration_min / 60.0)
        assert 8.0 < speed.median() < 25.0

    def test_passenger_counts_in_range(self, taxi):
        assert taxi.passenger_count.between(1, 6).all()

    def test_trip_lengths_are_right_skewed(self, taxi):
        """Real urban trip distributions have a heavy right tail."""
        assert taxi.distance_km.mean() > taxi.distance_km.median()


class TestEmbeddedStructure:
    def test_rush_hour_is_slower(self, taxi):
        """The congestion signal the duration model is supposed to learn."""
        speed = taxi.distance_km / (taxi.duration_min / 60.0)
        hour = taxi.pickup_datetime.dt.hour
        weekday = taxi.pickup_datetime.dt.dayofweek < 5
        rush = speed[weekday & hour.isin([8, 17, 18])].median()
        quiet = speed[weekday & hour.isin([2, 3, 4])].median()
        assert rush < quiet

    def test_longer_trips_run_faster(self, taxi):
        """Short trips are intersection-bound; long ones use arterials."""
        speed = taxi.distance_km / (taxi.duration_min / 60.0)
        short = speed[taxi.distance_km < 2].median()
        long = speed[taxi.distance_km > 8].median()
        assert long > short

    def test_electric_bikes_are_faster(self, bike):
        speed = bike.distance_km / (bike.duration_min / 60.0)
        electric = bike.rideable_type == "electric_bike"
        assert speed[electric].median() > speed[~electric].median()

    def test_fares_rise_with_distance(self, taxi):
        correlation = np.corrcoef(taxi.distance_km, taxi.fare_amount)[0, 1]
        assert correlation > 0.85


class TestReproducibility:
    def test_same_seed_gives_identical_data(self):
        a = generate_taxi_trips(n=500, seed=7)
        b = generate_taxi_trips(n=500, seed=7)
        assert a.equals(b)

    def test_different_seeds_diverge(self):
        a = generate_taxi_trips(n=500, seed=7)
        b = generate_taxi_trips(n=500, seed=8)
        assert not a.equals(b)
