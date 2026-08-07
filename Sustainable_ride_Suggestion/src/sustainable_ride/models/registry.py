"""Model loading and prediction at serving time.

A thin layer over the persisted bundle whose job is to hide from the
recommender whether a given estimate came from a trained model, a derivation,
or a fallback. It also holds the one piece of domain logic that could not be
learned: the scooter estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import resolve_path
from ..features.build import build_inference_row

logger = logging.getLogger(__name__)


class ModelsNotTrainedError(RuntimeError):
    """Raised when the model bundle is missing."""


@dataclass
class Prediction:
    """A single predicted quantity, with its provenance attached.

    ``source`` matters as much as ``value``: a consumer deserves to know
    whether a number was learned from 400,000 observed trips or derived from a
    speed assumption.
    """

    value: float
    source: str
    confidence: str = "medium"


class ModelRegistry:
    """Loads the trained bundle once and serves predictions from it."""

    _instance: "ModelRegistry | None" = None

    def __init__(self, bundle_path: Path | None = None) -> None:
        import joblib

        self.bundle_path = bundle_path or (resolve_path("models") / "models.joblib")
        if not self.bundle_path.exists():
            raise ModelsNotTrainedError(
                f"No trained models at {self.bundle_path}. Run:\n"
                f"    python -m sustainable_ride.cli train"
            )

        bundle = joblib.load(self.bundle_path)
        self.models = bundle["models"]
        self.scooter_calibration = bundle.get("scooter_calibration", {})
        self.sklearn_version = bundle.get("sklearn_version", "unknown")

        logger.info("Loaded %d models from %s (scikit-learn %s)",
                    len(self.models), self.bundle_path.name, self.sklearn_version)

    @classmethod
    def instance(cls) -> "ModelRegistry":
        """Process-wide singleton, so the API does not reload on every request."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # -- predictions -------------------------------------------------------

    def predict_taxi(self, origin, destination, when: pd.Timestamp,
                     passenger_count: int = 1) -> dict[str, Prediction]:
        """Predicted duration (minutes) and metered fare (USD) for a taxi trip."""
        X = build_inference_row(origin, destination, when, mode="taxi",
                                passenger_count=passenger_count)
        return {
            "duration_min": Prediction(
                value=float(self.models["taxi_duration"].predict(X)[0]),
                source="HistGradientBoosting trained on NYC TLC trip records",
                confidence="high",
            ),
            "fare_usd": Prediction(
                value=float(self.models["taxi_fare"].predict(X)[0]),
                source="HistGradientBoosting trained on NYC TLC metered fares",
                confidence="high",
            ),
        }

    def predict_bike(self, origin, destination, when: pd.Timestamp,
                     electric: bool = False) -> dict[str, Prediction]:
        """Predicted duration (minutes) for a bike trip."""
        X = build_inference_row(origin, destination, when, mode="bike",
                                is_electric=electric)
        return {
            "duration_min": Prediction(
                value=float(self.models["bike_duration"].predict(X)[0]),
                source="HistGradientBoosting trained on Citi Bike ride records",
                confidence="high",
            ),
        }

    def predict_scooter(self, origin, destination,
                        when: pd.Timestamp) -> dict[str, Prediction]:
        """Predicted duration (minutes) for a shared e-scooter trip.

        Derived, not learned. The e-bike model supplies the base estimate and a
        calibrated factor adjusts it. Flagged ``confidence="low"`` and given a
        source string that says plainly where the number comes from, so nothing
        downstream can mistake it for a trained prediction.
        """
        base = self.predict_bike(origin, destination, when, electric=True)
        factor = float(self.scooter_calibration.get("scooter_vs_ebike_adjustment", 0.98))

        return {
            "duration_min": Prediction(
                value=base["duration_min"].value / factor,
                source=(
                    "Derived from the e-bike model, adjusted by a factor of "
                    f"{factor} for a governed 15 mph scooter. No open shared-"
                    "scooter trip dataset exists to train against."
                ),
                confidence="low",
            ),
        }

    def metrics(self) -> dict:
        """Held-out test metrics for every trained model."""
        return {name: m.metrics for name, m in self.models.items()}

    def describe(self) -> dict:
        """Model card-ish summary, surfaced by the API's ``/models`` route."""
        return {
            "sklearn_version": self.sklearn_version,
            "models": {
                name: {
                    "target": m.target,
                    "estimator": type(m.estimator).__name__,
                    "log_target": m.log_target,
                    "train_rows": m.train_rows,
                    "data_source": m.data_source,
                    "n_features": len(m.feature_names),
                    "metrics": m.metrics,
                }
                for name, m in self.models.items()
            },
            "scooter_derivation": self.scooter_calibration,
        }
