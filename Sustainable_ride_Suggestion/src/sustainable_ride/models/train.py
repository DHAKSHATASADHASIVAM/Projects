"""Model training.

Three learned models:

===================  ====================================  =================
Model                Target                                Trained on
===================  ====================================  =================
``taxi_duration``    trip duration, minutes                TLC yellow cabs
``taxi_fare``        metered fare, USD                     TLC yellow cabs
``bike_duration``    trip duration, minutes                Citi Bike
===================  ====================================  =================

There is deliberately no scooter model. No open dataset of shared e-scooter
trips exists at anything like this scale, and inventing one would be worse than
useless -- it would look like evidence while being none. Instead the scooter
estimate is derived from the electric-bike model with a documented speed
adjustment, and the API labels it as derived. Saying "we do not have this data"
is a better answer than a confident fabrication.

``HistGradientBoostingRegressor`` is the estimator throughout. It handles the
263-level zone categoricals natively -- one-hot encoding them would produce a
526-column sparse matrix for no benefit -- trains on 400k rows in seconds, and
needs no feature scaling.

The duration models are trained on log1p(duration). Trip durations are strongly
right-skewed, and squared error on the raw scale would let a handful of hour-long
outliers dominate the loss at the expense of the 12-minute trips that make up
the bulk of the data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import load_config, random_seed, resolve_path
from ..features.build import (
    TAXI_CATEGORICAL_FEATURES,
    bike_feature_names,
    build_bike_features,
    build_taxi_features,
    taxi_feature_names,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted estimator plus everything needed to use and judge it."""

    name: str
    estimator: HistGradientBoostingRegressor
    feature_names: list[str]
    target: str
    log_target: bool
    metrics: dict[str, float] = field(default_factory=dict)
    train_rows: int = 0
    train_seconds: float = 0.0
    data_source: str = "unknown"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on the original scale, undoing any log transform."""
        X = X[self.feature_names]
        raw = self.estimator.predict(X)
        return np.expm1(raw) if self.log_target else raw


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics, including the ones that actually mean something here.

    MAPE is reported alongside MAE because the two answer different questions:
    a 3-minute error is trivial on a 40-minute trip and severe on a 4-minute
    one, and a mode recommender lives or dies on the short trips where the
    modes are genuinely competitive.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    nonzero = y_true > 1e-9

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero])
                                     / y_true[nonzero])) * 100.0),
        "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
        "p90_abs_error": float(np.percentile(np.abs(y_true - y_pred), 90)),
    }


def _fit(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    log_target: bool,
    categorical: list[str] | None,
    target_name: str,
    data_source: str,
) -> TrainedModel:
    cfg = load_config()
    seed = random_seed()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.get_path("training.test_size", 0.2)),
        random_state=seed,
    )

    # Flag the zone columns so the booster splits on them as unordered
    # categories rather than pretending zone 240 > zone 12.
    categorical_mask = None
    if categorical:
        categorical_mask = [c in categorical for c in X.columns]

    estimator = HistGradientBoostingRegressor(
        random_state=seed,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        **params,
    )

    y_train_fit = np.log1p(y_train) if log_target else y_train

    logger.info("Training %s on %s rows x %d features...",
                name, f"{len(X_train):,}", X_train.shape[1])
    started = time.perf_counter()
    estimator.fit(X_train, y_train_fit)
    elapsed = time.perf_counter() - started

    model = TrainedModel(
        name=name, estimator=estimator, feature_names=list(X.columns),
        target=target_name, log_target=log_target,
        train_rows=len(X_train), train_seconds=elapsed, data_source=data_source,
    )

    predictions = model.predict(X_test)
    model.metrics = _evaluate(y_test.to_numpy(), predictions)
    model.metrics["n_test"] = int(len(X_test))
    model.metrics["n_iterations"] = int(estimator.n_iter_)

    logger.info(
        "  %s: MAE %.2f | RMSE %.2f | R2 %.3f | MAPE %.1f%% (%.1fs, %d iters)",
        name, model.metrics["mae"], model.metrics["rmse"], model.metrics["r2"],
        model.metrics["mape"], elapsed, estimator.n_iter_,
    )
    return model


def train_taxi_models(taxi: pd.DataFrame,
                      data_source: str = "NYC TLC") -> dict[str, TrainedModel]:
    """Fit the taxi duration and fare models."""
    cfg = load_config()
    X = build_taxi_features(taxi)

    duration = _fit(
        "taxi_duration", X,
        pd.Series(taxi["duration_min"]).reset_index(drop=True),
        params=dict(cfg.get_path("training.models.taxi_duration", {})),
        log_target=True, categorical=TAXI_CATEGORICAL_FEATURES,
        target_name="duration_min", data_source=data_source,
    )

    fare = _fit(
        "taxi_fare", X,
        pd.Series(taxi["fare_amount"]).reset_index(drop=True),
        params=dict(cfg.get_path("training.models.taxi_fare", {})),
        log_target=True, categorical=TAXI_CATEGORICAL_FEATURES,
        target_name="fare_amount", data_source=data_source,
    )

    return {"taxi_duration": duration, "taxi_fare": fare}


def train_bike_model(bike: pd.DataFrame,
                     data_source: str = "Citi Bike") -> dict[str, TrainedModel]:
    """Fit the bike duration model, which also underpins scooter estimates."""
    cfg = load_config()
    X = build_bike_features(bike)

    duration = _fit(
        "bike_duration", X,
        pd.Series(bike["duration_min"]).reset_index(drop=True),
        params=dict(cfg.get_path("training.models.bike_duration", {})),
        log_target=True, categorical=None,
        target_name="duration_min", data_source=data_source,
    )
    return {"bike_duration": duration}


def calibrate_scooter_factor(bike: pd.DataFrame) -> dict:
    """Derive the scooter speed factor from observed e-bike vs classic speeds.

    Shared e-scooters are governed to roughly 15 mph (24 km/h), close to but a
    little below a Citi Bike e-bike, and they accelerate away from lights faster
    than a classic bike. Rather than assert a factor, we measure the observed
    e-bike/classic speed ratio in the Citi Bike data and apply a documented
    adjustment on top of it.

    This is an assumption, not a finding, and it is recorded as such so the
    number can be challenged.
    """
    speed = bike["distance_km"] / (bike["duration_min"] / 60.0)
    electric = bike["is_electric"].astype(bool)

    classic_speed = float(np.median(speed[~electric]))
    electric_speed = float(np.median(speed[electric]))
    observed_ratio = electric_speed / classic_speed if classic_speed > 0 else 1.0

    # Governed top speed is marginally below an e-bike's assisted cruise, but
    # scooters are unaffected by rider fitness. Net: treat them as ~2% slower.
    scooter_vs_ebike = 0.98

    return {
        "classic_median_kmh": round(classic_speed, 3),
        "electric_median_kmh": round(electric_speed, 3),
        "observed_electric_ratio": round(observed_ratio, 4),
        "scooter_vs_ebike_adjustment": scooter_vs_ebike,
        "scooter_speed_factor_vs_classic": round(observed_ratio * scooter_vs_ebike, 4),
        "basis": (
            "Measured from Citi Bike January 2024: median speed of electric "
            "vs classic bikes over the same city. The scooter adjustment on "
            "top is an assumption (governed 15 mph top speed, no rider-fitness "
            "variance), not an observation -- no open shared-scooter trip "
            "dataset exists to validate it against."
        ),
    }


def save_models(models: dict[str, TrainedModel], scooter_calibration: dict | None = None) -> Path:
    """Persist the fitted models and a machine-readable metrics summary."""
    import joblib

    model_dir = resolve_path("models")
    bundle_path = model_dir / "models.joblib"

    joblib.dump({
        "models": models,
        "scooter_calibration": scooter_calibration or {},
        "feature_names": {
            "taxi": taxi_feature_names(),
            "bike": bike_feature_names(),
        },
        "sklearn_version": __import__("sklearn").__version__,
    }, bundle_path, compress=3)

    summary = {
        name: {
            "target": m.target,
            "log_target": m.log_target,
            "train_rows": m.train_rows,
            "train_seconds": round(m.train_seconds, 2),
            "data_source": m.data_source,
            "n_features": len(m.feature_names),
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in m.metrics.items()},
        }
        for name, m in models.items()
    }
    if scooter_calibration:
        summary["scooter_derivation"] = scooter_calibration

    metrics_path = resolve_path("reports", "metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Saved %d models to %s (%.1f MB)", len(models), bundle_path,
                bundle_path.stat().st_size / 1e6)
    logger.info("Wrote metrics to %s", metrics_path)
    return bundle_path


def train_all(taxi_path: Path | None = None,
              bike_path: Path | None = None) -> dict[str, TrainedModel]:
    """Train every model from the processed datasets and persist them."""
    taxi_path = taxi_path or resolve_path("processed", "taxi_trips.parquet")
    bike_path = bike_path or resolve_path("processed", "bike_trips.parquet")

    if not taxi_path.exists() or not bike_path.exists():
        raise FileNotFoundError(
            f"Processed data not found ({taxi_path.name} / {bike_path.name}). "
            f"Run `python -m sustainable_ride.cli prepare` first."
        )

    quality_path = resolve_path("reports", "data_quality.json")
    source = "unknown"
    if quality_path.exists():
        try:
            entries = json.loads(quality_path.read_text(encoding="utf-8"))
            source = entries[0].get("source", "unknown") if entries else "unknown"
        except (json.JSONDecodeError, OSError, IndexError):
            pass

    taxi = pd.read_parquet(taxi_path)
    bike = pd.read_parquet(bike_path)

    models: dict[str, TrainedModel] = {}
    models.update(train_taxi_models(taxi, data_source=source))
    models.update(train_bike_model(bike, data_source=
                                   "Citi Bike" if source != "synthetic" else "synthetic"))

    calibration = calibrate_scooter_factor(bike)
    logger.info("Scooter speed factor vs classic bike: %.3f",
                calibration["scooter_speed_factor_vs_classic"])

    save_models(models, calibration)
    return models
