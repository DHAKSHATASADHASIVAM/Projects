"""Feature engineering.

The critical property under test is that training and inference build the same
features from the same inputs. Feature skew is silent -- nothing crashes, the
model just quietly gets worse -- so it needs an explicit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sustainable_ride.features.build import (
    bike_feature_names,
    build_bike_features,
    build_inference_row,
    build_taxi_features,
    spatial_features,
    taxi_feature_names,
    temporal_features,
)


@pytest.fixture
def taxi_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "pickup_datetime": pd.to_datetime(
            ["2024-01-15 08:30:00", "2024-01-15 23:15:00", "2024-01-13 14:00:00"]),
        "pu_lat": [40.7580, 40.7484, 40.7061],
        "pu_lon": [-73.9855, -73.9857, -73.9969],
        "do_lat": [40.7061, 40.7580, 40.7484],
        "do_lon": [-73.9969, -73.9855, -73.9857],
        "passenger_count": [1, 2, 3],
        "PULocationID": [230, 161, 87],
        "DOLocationID": [87, 230, 161],
    })


@pytest.fixture
def bike_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "started_at": pd.to_datetime(
            ["2024-01-15 08:30:00", "2024-01-15 23:15:00", "2024-01-13 14:00:00"]),
        "start_lat": [40.7580, 40.7484, 40.7061],
        "start_lng": [-73.9855, -73.9857, -73.9969],
        "end_lat": [40.7061, 40.7580, 40.7484],
        "end_lng": [-73.9969, -73.9855, -73.9857],
        "is_electric": [True, False, True],
    })


class TestTemporalFeatures:
    def test_cyclical_encoding_is_bounded(self):
        features = temporal_features(pd.date_range("2024-01-01", periods=48, freq="h"))
        for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            assert features[col].between(-1.0, 1.0).all()

    def test_midnight_and_23h_are_adjacent(self):
        """The whole point of cyclical encoding: 23:00 is next to 00:00."""
        features = temporal_features(pd.Series(pd.to_datetime(
            ["2024-01-15 23:00:00", "2024-01-16 00:00:00", "2024-01-15 12:00:00"])))
        near = np.hypot(features.hour_sin[0] - features.hour_sin[1],
                        features.hour_cos[0] - features.hour_cos[1])
        far = np.hypot(features.hour_sin[0] - features.hour_sin[2],
                       features.hour_cos[0] - features.hour_cos[2])
        assert near < far

    def test_weekend_flag(self):
        features = temporal_features(pd.Series(pd.to_datetime(
            ["2024-01-13", "2024-01-15"])))     # Saturday, Monday
        assert features.is_weekend[0] == 1
        assert features.is_weekend[1] == 0

    def test_rush_hour_flag(self):
        features = temporal_features(pd.Series(pd.to_datetime(
            ["2024-01-15 08:00:00", "2024-01-15 13:00:00"])))
        assert features.is_rush_hour[0] == 1
        assert features.is_rush_hour[1] == 0

    def test_night_flag(self):
        features = temporal_features(pd.Series(pd.to_datetime(
            ["2024-01-15 02:00:00", "2024-01-15 13:00:00"])))
        assert features.is_night[0] == 1
        assert features.is_night[1] == 0


class TestSpatialFeatures:
    def test_manhattan_dominates_straight_line(self):
        features = spatial_features([40.7580], [-73.9855], [40.7061], [-73.9969])
        assert features.manhattan_km[0] >= features.straight_km[0]

    def test_bearing_components_form_a_unit_vector(self):
        features = spatial_features([40.7580], [-73.9855], [40.7061], [-73.9969])
        magnitude = features.bearing_sin[0] ** 2 + features.bearing_cos[0] ** 2
        assert magnitude == pytest.approx(1.0)


class TestDesignMatrices:
    def test_taxi_columns_match_declared_names(self, taxi_frame):
        assert list(build_taxi_features(taxi_frame).columns) == taxi_feature_names()

    def test_bike_columns_match_declared_names(self, bike_frame):
        assert list(build_bike_features(bike_frame).columns) == bike_feature_names()

    def test_no_missing_values(self, taxi_frame, bike_frame):
        assert not build_taxi_features(taxi_frame).isna().any().any()
        assert not build_bike_features(bike_frame).isna().any().any()

    def test_row_count_preserved(self, taxi_frame):
        assert len(build_taxi_features(taxi_frame)) == len(taxi_frame)

    def test_missing_passenger_count_defaults_to_one(self, taxi_frame):
        taxi_frame.loc[0, "passenger_count"] = np.nan
        assert build_taxi_features(taxi_frame)["passenger_count"][0] == 1


class TestTrainInferenceParity:
    """Training and serving must produce identical features for identical input."""

    def test_taxi_parity(self, taxi_frame):
        batch = build_taxi_features(taxi_frame)
        single = build_inference_row(
            origin=(taxi_frame.pu_lat[0], taxi_frame.pu_lon[0]),
            destination=(taxi_frame.do_lat[0], taxi_frame.do_lon[0]),
            when=taxi_frame.pickup_datetime[0], mode="taxi",
            passenger_count=int(taxi_frame.passenger_count[0]),
            pu_zone=int(taxi_frame.PULocationID[0]),
            do_zone=int(taxi_frame.DOLocationID[0]),
        )
        assert list(single.columns) == list(batch.columns)
        for col in batch.columns:
            assert single[col].iloc[0] == pytest.approx(batch[col].iloc[0]), col

    def test_bike_parity(self, bike_frame):
        batch = build_bike_features(bike_frame)
        single = build_inference_row(
            origin=(bike_frame.start_lat[0], bike_frame.start_lng[0]),
            destination=(bike_frame.end_lat[0], bike_frame.end_lng[0]),
            when=bike_frame.started_at[0], mode="bike",
            is_electric=bool(bike_frame.is_electric[0]),
        )
        assert list(single.columns) == list(batch.columns)
        for col in batch.columns:
            assert single[col].iloc[0] == pytest.approx(batch[col].iloc[0]), col
