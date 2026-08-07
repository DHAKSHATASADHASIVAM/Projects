"""CO2-equivalent estimation for each travel mode.

The arithmetic here is deliberately simple; the substance is in which terms are
included. Two choices drive almost the whole result:

*Lifecycle vs operational.* A shared e-scooter has no tailpipe, so on an
operational basis it looks nearly carbon-free. Include the manufacturing burden
amortised over a short service life, plus the vans that collect and redistribute
the fleet, and it lands at roughly a quarter of a taxi's footprint rather than
a fortieth. The literature is unambiguous that the operational-only comparison
is misleading, so ``lifecycle`` is the default basis.

*Deadheading.* A taxi's revenue kilometre is not its only kilometre. Cruising
for the next fare, and driving to the pickup, add roughly 45% on top. Mode
comparisons that omit this systematically flatter taxis.

See ``config/emissions.yaml`` for the numbers and their citations.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import load_emissions_config


@dataclass(frozen=True)
class EmissionResult:
    """CO2e attributable to one trip by one mode."""

    mode: str
    co2_grams: float
    basis: str
    factor_g_per_km: float
    deadhead_multiplier: float
    occupancy: float
    effective_km: float
    """Distance the footprint is charged against, after the deadhead uplift."""

    @property
    def co2_kg(self) -> float:
        return self.co2_grams / 1000.0


def emission_factor(mode: str, basis: str | None = None) -> float:
    """Grams of CO2e per kilometre for ``mode`` on the given accounting basis."""
    cfg = load_emissions_config()
    basis = basis or cfg.get_path("basis", "lifecycle")
    if basis not in ("operational", "lifecycle"):
        raise ValueError(f"basis must be 'operational' or 'lifecycle', got {basis!r}")

    factors = cfg.get_path(f"factors.{mode}")
    if factors is None:
        raise KeyError(
            f"No emission factor configured for mode {mode!r}. "
            f"Known modes: {sorted((cfg.get_path('factors') or {}).keys())}"
        )
    return float(factors[f"{basis}_g_per_km"])


def estimate_co2(
    mode: str,
    distance_km: float,
    occupancy: float | None = None,
    basis: str | None = None,
) -> EmissionResult:
    """Estimate CO2e for a trip of ``distance_km`` by ``mode``.

    ``occupancy`` splits a shared vehicle's footprint across its passengers;
    only meaningful for taxis.
    """
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")

    cfg = load_emissions_config()
    basis = basis or cfg.get_path("basis", "lifecycle")

    factor = emission_factor(mode, basis)
    deadhead = float(cfg.get_path(f"deadhead_multiplier.{mode}", 1.0))

    if occupancy is None:
        occupancy = float(cfg.get_path(f"occupancy.{mode}",
                                       cfg.get_path(f"occupancy.{mode}_default", 1.0)))
    occupancy = max(1.0, float(occupancy))

    effective_km = distance_km * deadhead
    grams = (effective_km * factor) / occupancy

    return EmissionResult(
        mode=mode,
        co2_grams=grams,
        basis=basis,
        factor_g_per_km=factor,
        deadhead_multiplier=deadhead,
        occupancy=occupancy,
        effective_km=effective_km,
    )


def co2_saved_vs(baseline_mode: str, chosen: EmissionResult,
                 baseline: EmissionResult) -> float:
    """Grams of CO2e avoided by taking ``chosen`` instead of ``baseline_mode``.

    Negative when the chosen mode is in fact the dirtier one -- worth surfacing
    honestly rather than clamping to zero.
    """
    return baseline.co2_grams - chosen.co2_grams


def humanise_co2(grams: float) -> str:
    """A plain-language equivalence for a CO2 quantity.

    Abstract gram counts do not motivate anyone. Anchoring to something
    physical -- phone charges, days of a tree's work -- is what makes a
    sustainability figure land.
    """
    cfg = load_emissions_config()
    per_charge = float(cfg.get_path("equivalences.smartphone_charge_g", 8.2))
    per_tree_day = float(cfg.get_path("equivalences.tree_absorption_g_per_day", 60.0))

    grams = abs(float(grams))
    if grams < 1.0:
        return "negligible"
    if grams < 500.0:
        return f"{grams / per_charge:.0f} smartphone charges"
    return f"{grams / per_tree_day:.1f} days of a mature tree's CO2 absorption"


def factor_provenance() -> list[dict]:
    """The emission factors with their citations, for the API and the UI.

    Exposing the sources alongside the numbers is the point: a sustainability
    claim that cannot be traced back to a reference is not worth much.
    """
    cfg = load_emissions_config()
    basis = cfg.get_path("basis", "lifecycle")
    rows = []
    for mode, spec in (cfg.get_path("factors") or {}).items():
        rows.append({
            "mode": mode,
            "basis": basis,
            "g_per_km": float(spec[f"{basis}_g_per_km"]),
            "operational_g_per_km": float(spec["operational_g_per_km"]),
            "lifecycle_g_per_km": float(spec["lifecycle_g_per_km"]),
            "deadhead_multiplier": float(cfg.get_path(f"deadhead_multiplier.{mode}", 1.0)),
            "source": " ".join(str(spec.get("source", "")).split()),
            "notes": " ".join(str(spec.get("notes", "")).split()),
        })
    return rows
