"""Synthetic trip generator -- the fallback when the real data is unavailable.

This exists so the repository clones and runs end-to-end without a 420 MB
download. It is *not* a substitute for the real thing, and the README says so:
models trained on synthetic data can only recover the structure that was
deliberately written into the generator, which makes their accuracy scores
circular. The real TLC and Citi Bike pipelines are the ones that count.

What is modelled, because these are the effects the recommender depends on:

* A diurnal congestion profile -- speeds collapse during the morning and
  evening peaks and recover overnight.
* Distance-dependent speed: short trips are dominated by intersections and
  average far lower speeds than long ones, which is why a naive constant-speed
  estimate is badly wrong at both ends of the range.
* Weekday/weekend divergence, which shifts the peaks rather than removing them.
* Lognormal trip-length distribution, matching the heavy right tail of real
  urban trip data.
* Fares generated from the actual tariff plus multiplicative noise, so the fare
  model has a genuine, if easy, signal to recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import random_seed
from ..features.geo import haversine_km
from ..pricing import taxi_fare_analytic

# A handful of real NYC anchor points, so synthetic coordinates land on the
# city rather than in the abstract.
_ANCHORS = np.array([
    [40.7580, -73.9855],   # Times Square
    [40.7484, -73.9857],   # Empire State
    [40.7061, -73.9969],   # Financial District
    [40.7794, -73.9632],   # Upper East Side
    [40.7831, -73.9712],   # Upper West Side
    [40.7061, -73.9442],   # Williamsburg
    [40.6782, -73.9442],   # Bed-Stuy
    [40.7282, -73.7949],   # Flushing
    [40.6413, -73.7781],   # JFK
    [40.7769, -73.8740],   # LaGuardia
    [40.6892, -74.0445],   # Statue of Liberty area
    [40.8296, -73.9262],   # South Bronx
])


def _congestion_multiplier(hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    """Travel-time multiplier by hour of day. 1.0 is free-flow."""
    # Two Gaussian humps for the weekday peaks; weekends get one broad,
    # shallower midday hump instead.
    weekday = (
        1.00
        + 0.55 * np.exp(-0.5 * ((hour - 8.5) / 1.6) ** 2)
        + 0.70 * np.exp(-0.5 * ((hour - 17.8) / 2.0) ** 2)
        - 0.22 * np.exp(-0.5 * ((hour - 3.5) / 2.2) ** 2)
    )
    weekend = (
        1.00
        + 0.30 * np.exp(-0.5 * ((hour - 14.5) / 3.5) ** 2)
        - 0.25 * np.exp(-0.5 * ((hour - 4.0) / 2.5) ** 2)
    )
    return np.where(is_weekend, weekend, weekday)


def _sample_od(rng: np.random.Generator, n: int):
    """Sample origin/destination pairs clustered around the anchor points."""
    o_idx = rng.integers(0, len(_ANCHORS), n)
    d_idx = rng.integers(0, len(_ANCHORS), n)
    # Resample degenerate pairs so we do not generate zero-length trips.
    same = o_idx == d_idx
    while same.any():
        d_idx[same] = rng.integers(0, len(_ANCHORS), same.sum())
        same = o_idx == d_idx

    spread = 0.018  # ~2 km of jitter around each anchor
    o = _ANCHORS[o_idx] + rng.normal(0, spread, (n, 2))
    d = _ANCHORS[d_idx] + rng.normal(0, spread, (n, 2))
    return o[:, 0], o[:, 1], d[:, 0], d[:, 1]


def _sample_timestamps(rng: np.random.Generator, n: int, month: str = "2024-01"):
    """Timestamps across one month, weighted toward realistic hours."""
    start = pd.Timestamp(f"{month}-01")
    days = rng.integers(0, 28, n)
    # Hour-of-day demand profile: low overnight, peaks morning and evening.
    hour_weights = np.array([
        1.2, 0.7, 0.5, 0.4, 0.4, 0.8, 2.0, 4.0, 5.5, 4.8, 4.0, 4.2,
        4.5, 4.4, 4.3, 4.6, 5.4, 6.5, 6.8, 5.8, 4.6, 3.8, 3.0, 2.0,
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(24, size=n, p=hour_weights)
    minutes = rng.integers(0, 60, n)
    return start + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h") \
        + pd.to_timedelta(minutes, "m")


def generate_taxi_trips(n: int | None = None, month: str = "2024-01",
                        seed: int | None = None) -> pd.DataFrame:
    """Synthetic taxi trips in the processed schema."""
    from ..config import load_config
    n = n or int(load_config().get_path("data.synthetic.n_taxi", 120000))
    rng = np.random.default_rng(seed if seed is not None else random_seed())

    pu_lat, pu_lon, do_lat, do_lon = _sample_od(rng, n)
    ts = _sample_timestamps(rng, n, month)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    is_weekend = dow >= 5

    straight_km = haversine_km(pu_lat, pu_lon, do_lat, do_lon)
    # Road distance: circuity with trip-specific variation.
    circuity = rng.normal(1.42, 0.11, n).clip(1.05, 2.2)
    distance_km = np.maximum(straight_km * circuity, 0.35)

    # Free-flow speed rises with trip length -- longer trips use arterials and
    # bridges rather than crawling through the grid.
    base_speed = 13.0 + 9.5 * np.log1p(distance_km)
    base_speed = np.clip(base_speed, 9.0, 45.0)
    speed = base_speed / _congestion_multiplier(hour, is_weekend)
    speed *= rng.lognormal(0.0, 0.17, n)          # driver/route variation
    speed = np.clip(speed, 3.5, 70.0)

    duration_min = (distance_km / speed) * 60.0
    duration_min += rng.exponential(1.4, n)        # pickup/dropoff friction
    duration_min = np.clip(duration_min, 1.0, 180.0)

    passenger_count = rng.choice([1, 2, 3, 4, 5, 6],
                                 p=[0.71, 0.16, 0.05, 0.04, 0.02, 0.02], size=n)

    # Fares from the real tariff, vectorised approximately, then jittered.
    fare = np.array([
        taxi_fare_analytic(
            distance_km=float(d), duration_min=float(t),
            hour=int(h), is_weekday=not bool(w),
            origin=(float(a), float(b)), destination=(float(c), float(e)),
        ).total_usd
        for d, t, h, w, a, b, c, e in zip(
            distance_km, duration_min, hour, is_weekend,
            pu_lat, pu_lon, do_lat, do_lon)
    ])
    fare *= rng.lognormal(0.0, 0.05, n)            # meter/route noise

    # Synthetic zone ids: nearest anchor stands in for a taxi zone.
    pu_zone = _nearest_anchor(pu_lat, pu_lon) + 1
    do_zone = _nearest_anchor(do_lat, do_lon) + 1

    return pd.DataFrame({
        "pickup_datetime": ts,
        "pu_lat": pu_lat, "pu_lon": pu_lon,
        "do_lat": do_lat, "do_lon": do_lon,
        "PULocationID": pu_zone, "DOLocationID": do_zone,
        "passenger_count": passenger_count,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "fare_amount": np.round(fare, 2),
    })


def generate_bike_trips(n: int | None = None, month: str = "2024-01",
                        seed: int | None = None) -> pd.DataFrame:
    """Synthetic bike trips in the processed schema."""
    from ..config import load_config
    n = n or int(load_config().get_path("data.synthetic.n_bike", 120000))
    rng = np.random.default_rng((seed if seed is not None else random_seed()) + 1)

    s_lat, s_lon, e_lat, e_lon = _sample_od(rng, n)
    ts = _sample_timestamps(rng, n, month)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    is_weekend = dow >= 5

    straight_km = haversine_km(s_lat, s_lon, e_lat, e_lon)
    circuity = rng.normal(1.28, 0.10, n).clip(1.02, 2.0)
    distance_km = np.maximum(straight_km * circuity, 0.25)

    electric = rng.random(n) < 0.38
    # Cyclists are far less exposed to congestion than drivers -- the peak
    # penalty is real (crowded lanes, more signals honoured) but much smaller.
    congestion = 1.0 + 0.25 * (_congestion_multiplier(hour, is_weekend) - 1.0)
    base_speed = np.where(electric, 18.5, 14.2)
    speed = base_speed / congestion
    speed *= rng.lognormal(0.0, 0.20, n)           # rider fitness spread
    speed = np.clip(speed, 4.0, 32.0)

    duration_min = (distance_km / speed) * 60.0
    duration_min += rng.exponential(0.8, n)        # docking friction
    duration_min = np.clip(duration_min, 1.0, 120.0)

    return pd.DataFrame({
        "started_at": ts,
        "start_lat": s_lat, "start_lng": s_lon,
        "end_lat": e_lat, "end_lng": e_lon,
        "rideable_type": np.where(electric, "electric_bike", "classic_bike"),
        "distance_km": distance_km,
        "duration_min": duration_min,
    })


def _nearest_anchor(lat, lon) -> np.ndarray:
    """Index of the closest anchor point, used as a stand-in zone id."""
    lat = np.asarray(lat)[:, None]
    lon = np.asarray(lon)[:, None]
    d = haversine_km(lat, lon, _ANCHORS[None, :, 0], _ANCHORS[None, :, 1])
    return np.argmin(d, axis=1)


def generate_all(seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Both synthetic datasets, in the processed schema."""
    return {
        "taxi": generate_taxi_trips(seed=seed),
        "bike": generate_bike_trips(seed=seed),
    }
