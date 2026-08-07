"""Fare estimation.

Two independent estimators for taxi cost:

1. :func:`taxi_fare_analytic` reconstructs the metered fare from the published
   TLC tariff -- initial charge, per-fifth-mile units, slow-traffic time
   charging, and the stack of surcharges.
2. The gradient boosting model in :mod:`sustainable_ride.models`, trained on
   400k observed fares.

Neither is redundant. The tariff is exactly right about the rules and blind to
reality (it cannot know that this route crawls at 6 mph on a Tuesday evening);
the model has absorbed reality and is fuzzy about the rules. Their disagreement
is diagnostic, and ``reports/evaluation.json`` reports it: where the two diverge
is where the traffic model is doing real work.

Bike and scooter fares are purely mechanical -- they are published rate cards,
so there is nothing to learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import load_pricing_config


@dataclass(frozen=True)
class FareBreakdown:
    """An itemised fare, so a user can see where the money went."""

    mode: str
    total_usd: float
    base_usd: float = 0.0
    distance_usd: float = 0.0
    time_usd: float = 0.0
    surcharges: dict[str, float] = field(default_factory=dict)
    tip_usd: float = 0.0
    notes: str = ""

    @property
    def surcharge_total(self) -> float:
        return sum(self.surcharges.values())


def _is_congestion_zone(lat: float | None, lon: float | None) -> bool:
    """Whether a point is in the Manhattan congestion-surcharge zone.

    The real boundary is south of 96th Street in Manhattan. Approximated here
    with a latitude cut and a longitude band -- adequate for a fare estimate,
    and flagged as approximate rather than quietly presented as exact.
    """
    if lat is None or lon is None:
        return False
    return (40.70 <= lat <= 40.79) and (-74.02 <= lon <= -73.93)


def taxi_fare_analytic(
    distance_km: float,
    duration_min: float,
    hour: int,
    is_weekday: bool,
    origin: tuple[float, float] | None = None,
    destination: tuple[float, float] | None = None,
    include_tip: bool | None = None,
) -> FareBreakdown:
    """Reconstruct a NYC yellow-cab fare from the published tariff."""
    cfg = load_pricing_config()
    t = cfg.get_path("taxi")

    distance_miles = max(0.0, distance_km) / 1.609344
    duration_min = max(0.0, duration_min)

    base = float(t["initial_charge"])

    # The meter charges distance when moving above 12 mph and time when below.
    # We do not have a second-by-second speed trace, so we split the trip by
    # its average speed -- the standard approximation, and the reason the
    # analytic estimate degrades in heavy congestion.
    avg_speed_mph = (distance_miles / (duration_min / 60.0)) if duration_min > 0 else 0.0
    slow_threshold = float(t["slow_traffic_speed_mph"])

    if avg_speed_mph >= slow_threshold or duration_min == 0:
        units = math.ceil(distance_miles / float(t["unit_miles"])) if distance_miles > 0 else 0
        distance_charge = units * float(t["per_unit_charge"])
        time_charge = 0.0
    else:
        # Below the threshold the meter ticks on time. Charge the distance the
        # cab would have covered at threshold speed, and the remaining time.
        moving_miles = min(distance_miles, (duration_min / 60.0) * slow_threshold)
        units = math.ceil(moving_miles / float(t["unit_miles"])) if moving_miles > 0 else 0
        distance_charge = units * float(t["per_unit_charge"])
        stopped_min = max(0.0, duration_min - (moving_miles / slow_threshold) * 60.0)
        time_charge = stopped_min * float(t["per_minute_slow_traffic"])

    s = t["surcharges"]
    surcharges: dict[str, float] = {
        "mta_state": float(s["mta_state"]),
        "improvement": float(s["improvement"]),
    }
    if hour in set(t["night_hours"]):
        surcharges["night"] = float(s["night"])
    if is_weekday and hour in set(t["peak_hours"]):
        surcharges["weekday_peak"] = float(s["weekday_peak"])
    if _is_congestion_zone(*(origin or (None, None))) or \
       _is_congestion_zone(*(destination or (None, None))):
        surcharges["congestion"] = float(s["congestion"])

    subtotal = base + distance_charge + time_charge + sum(surcharges.values())

    if include_tip is None:
        include_tip = bool(t.get("include_tip_in_total", False))
    tip = subtotal * float(t["tip_rate"]) if include_tip else 0.0

    return FareBreakdown(
        mode="taxi",
        total_usd=round(subtotal + tip, 2),
        base_usd=base,
        distance_usd=round(distance_charge, 2),
        time_usd=round(time_charge, 2),
        surcharges={k: round(v, 2) for k, v in surcharges.items()},
        tip_usd=round(tip, 2),
        notes=(
            "Metered fare reconstructed from the TLC tariff. Congestion zone is "
            "approximated by a bounding box; tolls are not included."
        ),
    )


def bike_fare(duration_min: float, electric: bool = False,
              plan: str | None = None) -> FareBreakdown:
    """Citi Bike cost for a ride of the given duration."""
    cfg = load_pricing_config()
    b = cfg.get_path("bike")
    plan = plan or b.get("default_plan", "single_ride")
    duration_min = max(0.0, duration_min)

    if plan == "member":
        m = b["membership"]
        if electric:
            total = duration_min * float(m["electric_per_minute"])
            return FareBreakdown(
                mode="bike", total_usd=round(total, 2),
                time_usd=round(total, 2),
                notes="Member e-bike rate; annual membership cost not amortised in.",
            )
        included = float(m["classic_included_minutes"])
        overage = max(0.0, duration_min - included) * float(m["classic_overage_per_minute"])
        return FareBreakdown(
            mode="bike", total_usd=round(overage, 2), time_usd=round(overage, 2),
            notes=f"Member classic rate; first {included:.0f} min included.",
        )

    tier = b["electric" if electric else "classic"]
    unlock = float(tier["unlock_fee"])
    included = float(tier["included_minutes"])
    overage = max(0.0, duration_min - included) * float(tier["overage_per_minute"])

    return FareBreakdown(
        mode="bike",
        total_usd=round(unlock + overage, 2),
        base_usd=unlock,
        time_usd=round(overage, 2),
        notes=(
            f"Single-ride {'e-bike' if electric else 'classic'} pricing; "
            f"first {included:.0f} min included."
        ),
    )


def scooter_fare(duration_min: float) -> FareBreakdown:
    """Shared e-scooter cost: unlock fee plus per-minute rate."""
    cfg = load_pricing_config()
    s = cfg.get_path("scooter")
    duration_min = max(0.0, duration_min)

    unlock = float(s["unlock_fee"])
    time_charge = duration_min * float(s["per_minute"])
    total = max(float(s.get("minimum_fare", 0.0)), unlock + time_charge)

    return FareBreakdown(
        mode="scooter",
        total_usd=round(total, 2),
        base_usd=unlock,
        time_usd=round(time_charge, 2),
        notes=(
            "Representative NYC-area operator pricing. Operators do not publish "
            "rate cards, so this is the least certain cost estimate here."
        ),
    )
