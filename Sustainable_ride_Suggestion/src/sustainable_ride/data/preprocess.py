"""Cleaning and normalisation of the raw trip records into a common schema.

Public trip data is dirty in ways that matter. In the January 2024 TLC extract
there are trips with zero passengers, negative fares (refunds recorded as
trips), dropoff timestamps preceding pickups, 300-mile "trips" inside a borough,
and rides where the meter recorded distance but the clock recorded none. Left
in, these do not merely add noise -- they bias the models, because the errors
are not symmetric. A regressor trained on uncleaned fares will happily learn
that some trips cost negative money.

Every filter applied here is counted and reported, so the cleaning is auditable
rather than a silent black box. The counts land in
``reports/data_quality.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..features.geo import haversine_km, within_nyc

logger = logging.getLogger(__name__)

MILES_TO_KM = 1.609344


@dataclass
class CleaningReport:
    """Row counts at each stage of the filtering, for auditability."""

    dataset: str
    initial_rows: int = 0
    cleaned_rows: int = 0
    """Rows surviving the quality filters, before any training subsample."""

    final_rows: int = 0
    """Rows actually written out, after subsampling."""

    dropped: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str, before: int, after: int) -> None:
        if before != after:
            self.dropped[reason] = self.dropped.get(reason, 0) + (before - after)

    @property
    def retention_pct(self) -> float:
        """Share of raw rows that passed the quality filters.

        Deliberately measured against ``cleaned_rows``, not ``final_rows``:
        subsampling for training speed is not data loss, and folding it into
        the retention figure would misrepresent how dirty the source is.
        """
        return 100.0 * self.cleaned_rows / self.initial_rows if self.initial_rows else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "initial_rows": self.initial_rows,
            "cleaned_rows": self.cleaned_rows,
            "final_rows": self.final_rows,
            "retention_pct": round(self.retention_pct, 2),
            "subsampled": self.final_rows < self.cleaned_rows,
            "dropped": dict(sorted(self.dropped.items(), key=lambda kv: -kv[1])),
        }

    def log(self) -> None:
        logger.info("[%s] %s raw -> %s cleaned (%.1f%% retained)", self.dataset,
                    f"{self.initial_rows:,}", f"{self.cleaned_rows:,}", self.retention_pct)
        for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            logger.info("    -%9s  %s", f"{count:,}", reason)
        if self.final_rows < self.cleaned_rows:
            logger.info("    subsampled to %s rows for training",
                        f"{self.final_rows:,}")


def load_zone_centroids() -> pd.DataFrame:
    """Zone id -> centroid coordinates, building the table if necessary."""
    path = resolve_path("reference", "taxi_zone_centroids.csv")
    if not path.exists():
        from .download import build_zone_centroids
        return build_zone_centroids()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Taxi
# ---------------------------------------------------------------------------

def clean_taxi(df: pd.DataFrame, sample_size: int | None = None,
               seed: int = 42) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean TLC yellow taxi records into the common trip schema."""
    cfg = load_config()
    limits = cfg.get_path("cleaning.taxi")
    report = CleaningReport(dataset="taxi", initial_rows=len(df))

    df = df.copy()
    df = df.rename(columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    })

    # Standard rate code only. Codes 2-6 are JFK/Newark flat fares, negotiated
    # fares and group rides, whose prices are set by rule rather than by the
    # meter. Including them would teach the fare model that some long trips
    # cost a flat $70 regardless of distance -- true, but not a function of the
    # features we have.
    before = len(df)
    df = df[df["RatecodeID"] == 1]
    report.record("non-standard rate code (airport flat / negotiated)", before, len(df))

    df["duration_min"] = (
        (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    )
    df["distance_km"] = df["trip_distance"] * MILES_TO_KM

    before = len(df)
    df = df[df["duration_min"].between(limits["min_duration_min"],
                                       limits["max_duration_min"])]
    report.record("implausible duration", before, len(df))

    before = len(df)
    df = df[df["distance_km"].between(limits["min_distance_km"],
                                      limits["max_distance_km"])]
    report.record("implausible metered distance", before, len(df))

    before = len(df)
    df = df[df["fare_amount"].between(limits["min_fare_usd"], limits["max_fare_usd"])]
    report.record("implausible fare (incl. negative refunds)", before, len(df))

    speed_kmh = df["distance_km"] / (df["duration_min"] / 60.0)
    before = len(df)
    df = df[speed_kmh.between(limits["min_speed_kmh"], limits["max_speed_kmh"])]
    report.record("implausible average speed", before, len(df))

    before = len(df)
    df = df[df["passenger_count"].fillna(1).between(1, 6)]
    report.record("passenger count outside 1-6", before, len(df))

    # Attach zone centroids. Trips that start and end in the same zone are
    # dropped: their straight-line distance is zero by construction, which
    # makes them useless for any distance-based feature and poisons the
    # circuity fit.
    before = len(df)
    df = df[df["PULocationID"] != df["DOLocationID"]]
    report.record("same pickup and dropoff zone", before, len(df))

    centroids = load_zone_centroids()[["LocationID", "latitude", "longitude", "borough"]]
    df = df.merge(
        centroids.rename(columns={"latitude": "pu_lat", "longitude": "pu_lon",
                                  "borough": "pu_borough"}),
        left_on="PULocationID", right_on="LocationID", how="left").drop(columns="LocationID")
    df = df.merge(
        centroids.rename(columns={"latitude": "do_lat", "longitude": "do_lon",
                                  "borough": "do_borough"}),
        left_on="DOLocationID", right_on="LocationID", how="left").drop(columns="LocationID")

    before = len(df)
    df = df.dropna(subset=["pu_lat", "pu_lon", "do_lat", "do_lon"])
    report.record("unmatched zone id (incl. 264/265 'Unknown')", before, len(df))

    before = len(df)
    df = df[within_nyc(df["pu_lat"], df["pu_lon"]) & within_nyc(df["do_lat"], df["do_lon"])]
    report.record("centroid outside NYC bounds", before, len(df))

    # Straight-line distance, and the circuity implied by the meter.
    df["straight_km"] = haversine_km(df["pu_lat"], df["pu_lon"],
                                     df["do_lat"], df["do_lon"])
    before = len(df)
    df = df[df["straight_km"] > 0.15]
    report.record("degenerate straight-line distance", before, len(df))

    df["circuity"] = df["distance_km"] / df["straight_km"]
    # A metered distance shorter than the straight line, or more than four
    # times it, means the zone centroid is a poor proxy for the true endpoint.
    before = len(df)
    df = df[df["circuity"].between(0.8, 4.0)]
    report.record("centroid/meter distance inconsistent", before, len(df))

    keep = [
        "pickup_datetime", "dropoff_datetime", "passenger_count",
        "PULocationID", "DOLocationID", "pu_borough", "do_borough",
        "pu_lat", "pu_lon", "do_lat", "do_lon",
        "distance_km", "straight_km", "circuity", "duration_min",
        "fare_amount", "total_amount", "tip_amount", "congestion_surcharge",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    df["passenger_count"] = df["passenger_count"].fillna(1).astype("int16")

    report.cleaned_rows = len(df)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    report.final_rows = len(df)
    report.log()
    return df, report


# ---------------------------------------------------------------------------
# Bike
# ---------------------------------------------------------------------------

def clean_bike(df: pd.DataFrame, sample_size: int | None = None,
               seed: int = 42) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean Citi Bike records into the common trip schema."""
    cfg = load_config()
    limits = cfg.get_path("cleaning.bike")
    report = CleaningReport(dataset="bike", initial_rows=len(df))

    df = df.copy()
    for col in ("started_at", "ended_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    before = len(df)
    df = df.dropna(subset=["started_at", "ended_at", "start_lat", "start_lng",
                           "end_lat", "end_lng"])
    report.record("missing timestamp or coordinate", before, len(df))

    df["duration_min"] = (
        (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
    )

    before = len(df)
    df = df[df["duration_min"].between(limits["min_duration_min"],
                                       limits["max_duration_min"])]
    report.record("implausible duration (incl. undocked bikes)", before, len(df))

    before = len(df)
    df = df[within_nyc(df["start_lat"], df["start_lng"])
            & within_nyc(df["end_lat"], df["end_lng"])]
    report.record("coordinate outside NYC bounds", before, len(df))

    df["straight_km"] = haversine_km(df["start_lat"], df["start_lng"],
                                     df["end_lat"], df["end_lng"])

    # Round trips -- returned to the same dock -- are a real and common ride
    # type, but their straight-line distance is zero, so they carry no usable
    # signal for a distance-to-duration model.
    before = len(df)
    df = df[df["straight_km"] >= limits["min_distance_km"]]
    report.record("round trip / negligible displacement", before, len(df))

    # Citi Bike gives no odometer reading, so road distance must be inferred.
    circuity = float(cfg.get_path("routing.circuity_defaults.bike", 1.28))
    df["distance_km"] = df["straight_km"] * circuity

    before = len(df)
    df = df[df["distance_km"] <= limits["max_distance_km"]]
    report.record("distance beyond micromobility range", before, len(df))

    speed_kmh = df["distance_km"] / (df["duration_min"] / 60.0)
    before = len(df)
    df = df[speed_kmh.between(limits["min_speed_kmh"], limits["max_speed_kmh"])]
    report.record("implausible average speed", before, len(df))

    df["is_electric"] = (df["rideable_type"].astype(str)
                         .str.contains("electric", case=False, na=False))
    df["is_member"] = df.get("member_casual", pd.Series("casual", index=df.index)) == "member"

    keep = [
        "started_at", "ended_at", "rideable_type", "is_electric", "is_member",
        "start_lat", "start_lng", "end_lat", "end_lng",
        "straight_km", "distance_km", "duration_min",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    report.cleaned_rows = len(df)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    report.final_rows = len(df)
    report.log()
    return df, report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_processed_datasets(use_synthetic: bool = False,
                             month: str | None = None) -> dict[str, Path]:
    """Produce ``data/processed/{taxi,bike}_trips.parquet``.

    Falls back to the synthetic generator when the real files are absent, so a
    fresh clone can run the whole pipeline without downloading anything.
    """
    cfg = load_config()
    month = month or cfg.get_path("data.month", "2024-01")
    seed = int(cfg.get_path("project.random_seed", 42))
    out: dict[str, Path] = {}
    reports: list[dict] = []

    taxi_raw = resolve_path("raw", f"yellow_tripdata_{month}.parquet")
    bike_raw = resolve_path("raw", f"citibike_{month.replace('-', '')}.zip")

    if use_synthetic or not (taxi_raw.exists() and bike_raw.exists()):
        if not use_synthetic:
            logger.warning(
                "Raw data not found (%s / %s). Falling back to the synthetic "
                "generator. Run `python -m sustainable_ride.cli download` for "
                "the real datasets -- results from synthetic data are "
                "illustrative only.",
                taxi_raw.name, bike_raw.name)
        from .synthetic import generate_bike_trips, generate_taxi_trips

        taxi = generate_taxi_trips(month=month, seed=seed)
        bike = generate_bike_trips(month=month, seed=seed)
        taxi["straight_km"] = haversine_km(taxi["pu_lat"], taxi["pu_lon"],
                                           taxi["do_lat"], taxi["do_lon"])
        taxi["circuity"] = taxi["distance_km"] / taxi["straight_km"]
        bike["straight_km"] = haversine_km(bike["start_lat"], bike["start_lng"],
                                           bike["end_lat"], bike["end_lng"])
        bike["is_electric"] = bike["rideable_type"] == "electric_bike"
        reports.append({"dataset": "taxi", "source": "synthetic", "final_rows": len(taxi)})
        reports.append({"dataset": "bike", "source": "synthetic", "final_rows": len(bike)})
    else:
        from .download import read_bike_zip

        logger.info("Reading raw taxi records from %s", taxi_raw.name)
        taxi, taxi_report = clean_taxi(
            pd.read_parquet(taxi_raw),
            sample_size=int(cfg.get_path("data.taxi.sample_size", 400000)), seed=seed)
        reports.append({**taxi_report.to_dict(), "source": "NYC TLC"})

        logger.info("Reading raw bike records from %s", bike_raw.name)
        bike, bike_report = clean_bike(
            read_bike_zip(bike_raw),
            sample_size=int(cfg.get_path("data.bike.sample_size", 400000)), seed=seed)
        reports.append({**bike_report.to_dict(), "source": "Citi Bike"})

    taxi_path = resolve_path("processed", "taxi_trips.parquet")
    bike_path = resolve_path("processed", "bike_trips.parquet")
    taxi.to_parquet(taxi_path, index=False)
    bike.to_parquet(bike_path, index=False)
    out["taxi"] = taxi_path
    out["bike"] = bike_path

    quality_path = resolve_path("reports", "data_quality.json")
    quality_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    logger.info("Wrote cleaning audit to %s", quality_path)

    return out


def fit_circuity(taxi: pd.DataFrame | None = None) -> dict:
    """Fit road-to-straight-line distance ratios from the metered taxi data.

    The taxi factor is measured directly: the TLC meter records the road
    distance actually driven, and we know the straight-line distance between
    zone centroids, so their ratio is observable over hundreds of thousands of
    trips. The median is used rather than the mean because the ratio
    distribution has a long right tail (detours, circling for parking).

    No equivalent odometer exists for bikes, so the cycling factor is scaled
    from the taxi one by the ratio reported in the literature -- bikes use
    two-way streets and paths that cars cannot, so their networks are less
    circuitous. That inference is documented rather than hidden.
    """
    if taxi is None:
        taxi = pd.read_parquet(resolve_path("processed", "taxi_trips.parquet"))

    if "circuity" not in taxi.columns:
        raise ValueError("Taxi frame has no `circuity` column; run cleaning first.")

    ratios = taxi["circuity"].to_numpy()
    ratios = ratios[np.isfinite(ratios)]
    taxi_factor = float(np.median(ratios))

    cfg = load_config()
    default_taxi = float(cfg.get_path("routing.circuity_defaults.taxi", 1.42))
    default_bike = float(cfg.get_path("routing.circuity_defaults.bike", 1.28))
    bike_factor = taxi_factor * (default_bike / default_taxi)

    result = {
        "factors": {
            "taxi": round(taxi_factor, 4),
            "bike": round(bike_factor, 4),
            "scooter": round(bike_factor, 4),
        },
        "n_trips": int(len(ratios)),
        "taxi_percentiles": {
            str(p): round(float(np.percentile(ratios, p)), 4)
            for p in (10, 25, 50, 75, 90)
        },
        "method": (
            "Taxi factor is the median of metered road distance divided by "
            "straight-line distance between zone centroids. Bike and scooter "
            "factors are scaled from it using the literature ratio, since no "
            "odometer reading exists for Citi Bike trips."
        ),
    }

    path = resolve_path("reference", "circuity.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Fitted circuity factors: %s", result["factors"])
    return result
