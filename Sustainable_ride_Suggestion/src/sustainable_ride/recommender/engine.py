"""The recommendation engine.

Assembles, for one origin-destination-time query:

1. A route per mode (live from OpenRouteService where available).
2. Predicted duration and cost, from the trained models plus the tariff modules.
3. Lifecycle CO2, from the emission factors.
4. Feasibility filtering -- distance limits, weather, night-time.
5. The Pareto frontier, then a weighted ranking within it.

Two design choices worth calling out.

*Door-to-door time, not vehicle time.* The models predict how long the vehicle
is moving. That is not what a traveller experiences. Waiting for a cab, walking
to a dock, and finding a parking spot at the other end are real minutes, and
they differ sharply by mode -- which means comparing raw vehicle times would
systematically favour whichever mode has the worst access overhead. The
``access_egress_min`` term in the config corrects for this.

*Infeasible options are returned, not dropped.* If a user asks about a 30 km
trip, "bike is not sensible here" is useful information. Silently returning a
single option would leave them wondering whether the system had even considered
the alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import load_config
from ..emissions import EmissionResult, estimate_co2, humanise_co2
from ..features.geo import validate_coordinate
from ..models.registry import ModelRegistry
from ..pricing import FareBreakdown, bike_fare, scooter_fare, taxi_fare_analytic
from ..routing import Route, get_router, profile_for_mode
from .pareto import pareto_frontier, weighted_scores

logger = logging.getLogger(__name__)

MODES = ("taxi", "bike", "scooter")


@dataclass
class ModeOption:
    """A fully-costed option for one travel mode."""

    mode: str
    label: str

    duration_min: float
    """Door-to-door, including access and egress."""

    vehicle_duration_min: float
    access_egress_min: float
    cost_usd: float
    co2_grams: float
    distance_km: float

    feasible: bool = True
    infeasible_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Provenance
    duration_source: str = ""
    duration_confidence: str = "medium"
    cost_source: str = ""
    route_provider: str = ""
    route_is_estimate: bool = True

    # Scoring, filled in by the engine
    score: float | None = None
    rank: int | None = None
    on_pareto_frontier: bool = False

    # Comparisons against the taxi baseline
    co2_saved_vs_taxi_g: float = 0.0
    cost_saved_vs_taxi_usd: float = 0.0
    time_delta_vs_taxi_min: float = 0.0

    fare_breakdown: dict = field(default_factory=dict)
    emission_detail: dict = field(default_factory=dict)
    geometry: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recommendation:
    """The engine's full answer to one query."""

    origin: tuple[float, float]
    destination: tuple[float, float]
    departure: datetime
    weights: dict[str, float]
    options: list[ModeOption]
    best: ModeOption | None
    frontier_modes: list[str]
    narrative: str = ""
    routing_provider: str = ""

    def to_dict(self) -> dict:
        return {
            "origin": {"lat": self.origin[0], "lon": self.origin[1]},
            "destination": {"lat": self.destination[0], "lon": self.destination[1]},
            "departure": self.departure.isoformat(),
            "weights": self.weights,
            "routing_provider": self.routing_provider,
            "best_mode": self.best.mode if self.best else None,
            "frontier_modes": self.frontier_modes,
            "narrative": self.narrative,
            "options": [o.to_dict() for o in self.options],
        }


class RideRecommender:
    """Ranks travel modes for a trip."""

    def __init__(self, registry: ModelRegistry | None = None, router=None) -> None:
        self.cfg = load_config()
        self.registry = registry or ModelRegistry.instance()
        self.router = router or get_router()

    # -- per-mode estimation ----------------------------------------------

    def _estimate_mode(self, mode: str, origin, destination,
                       when: pd.Timestamp, passenger_count: int,
                       include_geometry: bool) -> ModeOption:
        mode_cfg = self.cfg.get_path(f"modes.{mode}", {}) or {}
        route: Route = self.router.route(origin, destination, profile_for_mode(mode))

        if mode == "taxi":
            preds = self.registry.predict_taxi(origin, destination, when, passenger_count)
            vehicle_min = preds["duration_min"].value
            duration_source = preds["duration_min"].source
            confidence = preds["duration_min"].confidence

            model_fare = preds["fare_usd"].value
            tariff = taxi_fare_analytic(
                distance_km=route.distance_km, duration_min=vehicle_min,
                hour=int(when.hour), is_weekday=bool(when.dayofweek < 5),
                origin=origin, destination=destination,
            )
            # The learned fare captures metered reality; the tariff supplies the
            # surcharges, which are rules rather than patterns and which the
            # model cannot see (it never observed the congestion zone flag).
            cost = model_fare + tariff.surcharge_total
            cost_source = (
                "Learned metered fare (gradient boosting on TLC records) plus "
                "statutory surcharges computed from the published TLC tariff."
            )
            breakdown = {
                "learned_metered_fare": round(model_fare, 2),
                "tariff_only_estimate": tariff.total_usd,
                "surcharges": tariff.surcharges,
                "note": tariff.notes,
            }

        elif mode == "bike":
            preds = self.registry.predict_bike(origin, destination, when, electric=False)
            vehicle_min = preds["duration_min"].value
            duration_source = preds["duration_min"].source
            confidence = preds["duration_min"].confidence
            fare: FareBreakdown = bike_fare(vehicle_min, electric=False)
            cost = fare.total_usd
            cost_source = "Citi Bike published single-ride rate card."
            breakdown = {"unlock": fare.base_usd, "time_charge": fare.time_usd,
                         "note": fare.notes}

        elif mode == "scooter":
            preds = self.registry.predict_scooter(origin, destination, when)
            vehicle_min = preds["duration_min"].value
            duration_source = preds["duration_min"].source
            confidence = preds["duration_min"].confidence
            fare = scooter_fare(vehicle_min)
            cost = fare.total_usd
            cost_source = "Representative shared e-scooter pricing (unpublished; approximate)."
            breakdown = {"unlock": fare.base_usd, "time_charge": fare.time_usd,
                         "note": fare.notes}
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        occupancy = passenger_count if mode == "taxi" else 1
        emissions: EmissionResult = estimate_co2(mode, route.distance_km, occupancy=occupancy)

        access = float(mode_cfg.get("access_egress_min", 0.0))

        return ModeOption(
            mode=mode,
            label=str(mode_cfg.get("label", mode.title())),
            duration_min=round(vehicle_min + access, 1),
            vehicle_duration_min=round(vehicle_min, 1),
            access_egress_min=access,
            cost_usd=round(float(cost), 2),
            co2_grams=round(emissions.co2_grams, 1),
            distance_km=round(route.distance_km, 2),
            duration_source=duration_source,
            duration_confidence=confidence,
            cost_source=cost_source,
            route_provider=route.provider,
            route_is_estimate=route.is_estimate,
            fare_breakdown=breakdown,
            emission_detail={
                "basis": emissions.basis,
                "factor_g_per_km": emissions.factor_g_per_km,
                "deadhead_multiplier": emissions.deadhead_multiplier,
                "effective_km": round(emissions.effective_km, 2),
                "occupancy": emissions.occupancy,
            },
            geometry=[list(p) for p in route.geometry] if include_geometry else [],
        )

    # -- feasibility -------------------------------------------------------

    def _apply_feasibility(self, option: ModeOption, when: pd.Timestamp,
                           rain_probability: float,
                           accessibility_required: bool) -> None:
        mode_cfg = self.cfg.get_path(f"modes.{option.mode}", {}) or {}
        rec_cfg = self.cfg.get_path("recommender", {}) or {}

        max_km = float(mode_cfg.get("max_distance_km", 1e9))
        if option.distance_km > max_km:
            option.feasible = False
            option.infeasible_reasons.append(
                f"{option.distance_km:.1f} km exceeds the {max_km:.0f} km "
                f"practical range for this mode"
            )

        if mode_cfg.get("weather_sensitive", False):
            threshold = float(rec_cfg.get("rain_infeasible_threshold", 0.6))
            if rain_probability >= threshold:
                option.feasible = False
                option.infeasible_reasons.append(
                    f"{rain_probability:.0%} chance of rain makes an exposed "
                    f"mode impractical"
                )
            elif rain_probability >= 0.3:
                option.warnings.append(
                    f"{rain_probability:.0%} chance of rain -- you may get wet"
                )

            if int(when.hour) in set(rec_cfg.get("night_hours", [])):
                option.warnings.append(
                    "Night-time trip: reduced visibility and fewer available "
                    "vehicles nearby"
                )

        if accessibility_required and option.mode in ("bike", "scooter"):
            option.feasible = False
            option.infeasible_reasons.append(
                "Does not meet the stated step-free accessibility requirement"
            )

        if option.route_is_estimate:
            option.warnings.append(
                "Distance is a straight-line estimate scaled by a fitted "
                "circuity factor, not a measured road route"
            )

    # -- narrative ---------------------------------------------------------

    def _narrate(self, best: ModeOption | None, options: list[ModeOption],
                 weights: dict[str, float]) -> str:
        if best is None:
            return "No feasible travel option was found for this trip."

        taxi = next((o for o in options if o.mode == "taxi"), None)
        parts = [
            f"{best.label} is the best match for your priorities: "
            f"{best.duration_min:.0f} min door-to-door, ${best.cost_usd:.2f}, "
            f"{best.co2_grams:.0f} g CO2e."
        ]

        if taxi and best.mode != "taxi" and taxi.feasible:
            if best.co2_saved_vs_taxi_g > 0:
                parts.append(
                    f"That is {best.co2_saved_vs_taxi_g:.0f} g less CO2e than "
                    f"taking a taxi -- about {humanise_co2(best.co2_saved_vs_taxi_g)}."
                )
            if best.cost_saved_vs_taxi_usd > 0:
                parts.append(f"You also save ${best.cost_saved_vs_taxi_usd:.2f}.")
            if best.time_delta_vs_taxi_min > 1:
                parts.append(
                    f"It costs you {best.time_delta_vs_taxi_min:.0f} extra minutes."
                )
            elif best.time_delta_vs_taxi_min < -1:
                parts.append(
                    f"It is also {abs(best.time_delta_vs_taxi_min):.0f} minutes faster."
                )

        frontier = [o for o in options if o.on_pareto_frontier]
        if len(frontier) > 1:
            others = [o.label for o in frontier if o.mode != best.mode]
            parts.append(
                f"{', '.join(others)} {'are' if len(others) > 1 else 'is'} also "
                f"defensible -- no option beats them on every measure at once, so "
                f"this ranking reflects your stated weights, not a universal answer."
            )

        dominated = [o for o in options
                     if o.feasible and not o.on_pareto_frontier]
        if dominated:
            parts.append(
                f"{', '.join(o.label for o in dominated)} "
                f"{'are' if len(dominated) > 1 else 'is'} worse on every measure "
                f"than at least one alternative."
            )

        return " ".join(parts)

    # -- public API --------------------------------------------------------

    def recommend(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        departure: datetime | None = None,
        weights: dict[str, float] | None = None,
        passenger_count: int = 1,
        rain_probability: float = 0.0,
        accessibility_required: bool = False,
        include_geometry: bool = True,
        modes: tuple[str, ...] = MODES,
    ) -> Recommendation:
        """Rank travel modes for one trip."""
        validate_coordinate(origin[0], origin[1], "origin")
        validate_coordinate(destination[0], destination[1], "destination")

        if not 1 <= int(passenger_count) <= 6:
            raise ValueError("passenger_count must be between 1 and 6")
        if not 0.0 <= float(rain_probability) <= 1.0:
            raise ValueError("rain_probability must be between 0 and 1")

        departure = departure or datetime.now()
        when = pd.Timestamp(departure)
        weights = weights or dict(self.cfg.get_path("recommender.default_weights", {}))

        options = []
        for mode in modes:
            try:
                option = self._estimate_mode(mode, origin, destination, when,
                                             passenger_count, include_geometry)
            except Exception as exc:  # one broken mode must not sink the request
                logger.exception("Could not estimate mode %s: %s", mode, exc)
                continue
            self._apply_feasibility(option, when, rain_probability, accessibility_required)
            options.append(option)

        if not options:
            raise RuntimeError("No travel modes could be estimated for this trip")

        self._score(options, weights)

        taxi = next((o for o in options if o.mode == "taxi"), None)
        if taxi is not None:
            for option in options:
                option.co2_saved_vs_taxi_g = round(taxi.co2_grams - option.co2_grams, 1)
                option.cost_saved_vs_taxi_usd = round(taxi.cost_usd - option.cost_usd, 2)
                option.time_delta_vs_taxi_min = round(
                    option.duration_min - taxi.duration_min, 1)

        feasible = [o for o in options if o.feasible]
        best = min(feasible, key=lambda o: o.score) if feasible else None

        options.sort(key=lambda o: (not o.feasible,
                                    o.score if o.score is not None else 1e9))

        return Recommendation(
            origin=origin, destination=destination, departure=departure,
            weights=weights, options=options, best=best,
            frontier_modes=[o.mode for o in options if o.on_pareto_frontier],
            narrative=self._narrate(best, options, weights),
            routing_provider=getattr(self.router, "name", "unknown"),
        )

    def _score(self, options: list[ModeOption], weights: dict[str, float]) -> None:
        """Compute the Pareto frontier and the weighted ranking.

        Only feasible options take part. Including an infeasible mode would let
        it distort the min-max normalisation for everything else -- a 40 km bike
        ride would stretch the time axis and compress the real differences
        between the options a user can actually take.
        """
        feasible = [o for o in options if o.feasible]
        if not feasible:
            return

        objectives = [[o.cost_usd, o.duration_min, o.co2_grams] for o in feasible]
        for idx in pareto_frontier(objectives):
            feasible[idx].on_pareto_frontier = True

        scores = weighted_scores(
            cost=[o.cost_usd for o in feasible],
            time=[o.duration_min for o in feasible],
            co2=[o.co2_grams for o in feasible],
            weights=weights,
        )
        for option, score in zip(feasible, scores):
            option.score = round(float(score), 4)

        for rank, option in enumerate(sorted(feasible, key=lambda o: o.score), start=1):
            option.rank = rank
