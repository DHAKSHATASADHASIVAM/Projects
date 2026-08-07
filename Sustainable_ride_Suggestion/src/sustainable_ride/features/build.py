"""Feature construction, shared by training and inference.

The single most important property of this module is that training and serving
call *the same functions*. Feature skew -- where the training pipeline computes
a feature one way and the live service computes it another -- is the most common
way a model that scored well offline behaves badly in production, and it is
entirely avoidable by not writing the logic twice.

Cyclical encoding is used for hour-of-day and day-of-week. Feeding an hour in
as the integer 23 tells a tree-based model that 23:00 and 00:00 are maximally
distant, when they are adjacent. Projecting onto a sine/cosine pair restores
that adjacency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import bearing_deg, haversine_km, manhattan_km

# Feature groups, exported so the model modules and the evaluation report stay
# in agreement about what went in.
NUMERIC_FEATURES = [
    "straight_km", "manhattan_km", "bearing_sin", "bearing_cos",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush_hour", "is_night",
    "pu_lat", "pu_lon", "do_lat", "do_lon",
]

TAXI_CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]
TAXI_EXTRA_NUMERIC = ["passenger_count"]
BIKE_EXTRA_NUMERIC = ["is_electric"]

RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}


def temporal_features(timestamps: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical and categorical time-of-travel features."""
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype("int8"),
        "is_rush_hour": np.isin(hour, list(RUSH_HOURS)).astype("int8"),
        "is_night": np.isin(hour, list(NIGHT_HOURS)).astype("int8"),
    })


def spatial_features(pu_lat, pu_lon, do_lat, do_lon) -> pd.DataFrame:
    """Distance and direction features for an origin-destination pair."""
    pu_lat = np.asarray(pu_lat, dtype="float64")
    pu_lon = np.asarray(pu_lon, dtype="float64")
    do_lat = np.asarray(do_lat, dtype="float64")
    do_lon = np.asarray(do_lon, dtype="float64")

    straight = haversine_km(pu_lat, pu_lon, do_lat, do_lon)
    bearing = np.radians(bearing_deg(pu_lat, pu_lon, do_lat, do_lon))

    return pd.DataFrame({
        "straight_km": straight,
        "manhattan_km": manhattan_km(pu_lat, pu_lon, do_lat, do_lon),
        # Direction of travel matters in a gridded city: the avenues run fast
        # north-south, the cross streets slowly east-west.
        "bearing_sin": np.sin(bearing),
        "bearing_cos": np.cos(bearing),
        "pu_lat": pu_lat, "pu_lon": pu_lon,
        "do_lat": do_lat, "do_lon": do_lon,
    })


def build_taxi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the taxi model's design matrix."""
    features = pd.concat([
        spatial_features(df["pu_lat"], df["pu_lon"], df["do_lat"], df["do_lon"]),
        temporal_features(df["pickup_datetime"]),
    ], axis=1)

    features["passenger_count"] = (
        pd.Series(df["passenger_count"]).reset_index(drop=True)
        .fillna(1).astype("int16")
    )
    for col in TAXI_CATEGORICAL_FEATURES:
        features[col] = pd.Series(df[col]).reset_index(drop=True).astype("int32")

    return features[taxi_feature_names()]


def build_bike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the bike/scooter model's design matrix."""
    features = pd.concat([
        spatial_features(df["start_lat"], df["start_lng"],
                         df["end_lat"], df["end_lng"]),
        temporal_features(df["started_at"]),
    ], axis=1)

    features["is_electric"] = (
        pd.Series(df["is_electric"]).reset_index(drop=True).astype("int8")
    )
    return features[bike_feature_names()]


def taxi_feature_names() -> list[str]:
    return NUMERIC_FEATURES + TAXI_EXTRA_NUMERIC + TAXI_CATEGORICAL_FEATURES


def bike_feature_names() -> list[str]:
    return NUMERIC_FEATURES + BIKE_EXTRA_NUMERIC


def build_inference_row(
    origin: tuple[float, float],
    destination: tuple[float, float],
    when: pd.Timestamp,
    mode: str,
    passenger_count: int = 1,
    is_electric: bool = False,
    pu_zone: int | None = None,
    do_zone: int | None = None,
) -> pd.DataFrame:
    """Build a single-row design matrix for live prediction.

    Deliberately routed through the same ``spatial_features`` and
    ``temporal_features`` used in training.
    """
    spatial = spatial_features([origin[0]], [origin[1]],
                               [destination[0]], [destination[1]])
    temporal = temporal_features(pd.Series([pd.Timestamp(when)]))
    row = pd.concat([spatial, temporal], axis=1)

    if mode == "taxi":
        row["passenger_count"] = int(passenger_count)
        # Zone ids are optional at inference time -- a user drops a pin, they do
        # not know their TLC zone. When absent we resolve the nearest centroid;
        # the model treats unseen categories gracefully.
        row["PULocationID"] = int(pu_zone) if pu_zone is not None else nearest_zone(*origin)
        row["DOLocationID"] = int(do_zone) if do_zone is not None else nearest_zone(*destination)
        return row[taxi_feature_names()]

    row["is_electric"] = int(is_electric)
    return row[bike_feature_names()]


_ZONE_CACHE: pd.DataFrame | None = None


def nearest_zone(lat: float, lon: float) -> int:
    """TLC zone whose centroid is closest to a point.

    Uses the same centroid table the taxi model was trained against, so a live
    request is described in the same coordinate vocabulary as the training rows.
    """
    global _ZONE_CACHE
    if _ZONE_CACHE is None:
        from ..data.preprocess import load_zone_centroids
        _ZONE_CACHE = load_zone_centroids()

    distances = haversine_km(lat, lon,
                             _ZONE_CACHE["latitude"].to_numpy(),
                             _ZONE_CACHE["longitude"].to_numpy())
    return int(_ZONE_CACHE.iloc[int(np.argmin(distances))]["LocationID"])
