"""Emission estimation.

These tests encode the *relationships* that must hold between modes rather than
pinning exact gram counts. The factors are configurable and will change as
better sources appear; what must never change is that a taxi is dirtier than a
scooter, that a scooter is dirtier than a bike, and that the lifecycle basis is
never lower than the operational one.
"""

from __future__ import annotations

import pytest

from sustainable_ride.emissions import (
    emission_factor,
    estimate_co2,
    factor_provenance,
    humanise_co2,
)


class TestEmissionFactors:
    def test_ordering_across_modes(self):
        """Taxi > scooter > bike, on any accounting basis."""
        for basis in ("operational", "lifecycle"):
            taxi = emission_factor("taxi", basis)
            scooter = emission_factor("scooter", basis)
            bike = emission_factor("bike", basis)
            assert taxi > scooter >= bike, f"ordering violated on {basis} basis"

    def test_lifecycle_never_below_operational(self):
        """Manufacturing and fuel-chain emissions cannot be negative."""
        for mode in ("taxi", "bike", "scooter"):
            assert emission_factor(mode, "lifecycle") >= emission_factor(mode, "operational")

    def test_bike_has_no_tailpipe(self):
        assert emission_factor("bike", "operational") == 0.0

    def test_bike_lifecycle_is_not_zero(self):
        """A bike is not carbon-free once manufacturing is counted."""
        assert emission_factor("bike", "lifecycle") > 0.0

    def test_unknown_mode_raises(self):
        with pytest.raises(KeyError, match="No emission factor"):
            emission_factor("helicopter")

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError, match="operational.*lifecycle"):
            emission_factor("taxi", "vibes")


class TestEstimateCo2:
    def test_scales_linearly_with_distance(self):
        one = estimate_co2("taxi", 1.0).co2_grams
        ten = estimate_co2("taxi", 10.0).co2_grams
        assert ten == pytest.approx(one * 10.0)

    def test_zero_distance_is_zero(self):
        assert estimate_co2("bike", 0.0).co2_grams == 0.0

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            estimate_co2("taxi", -5.0)

    def test_deadhead_inflates_taxi_distance(self):
        """A taxi's footprint covers more km than the passenger travels."""
        result = estimate_co2("taxi", 10.0)
        assert result.deadhead_multiplier > 1.0
        assert result.effective_km > 10.0

    def test_bike_has_no_deadhead(self):
        result = estimate_co2("bike", 10.0)
        assert result.deadhead_multiplier == 1.0
        assert result.effective_km == pytest.approx(10.0)

    def test_occupancy_splits_taxi_footprint(self):
        """Four people sharing a cab each carry a quarter of the emissions."""
        solo = estimate_co2("taxi", 10.0, occupancy=1)
        shared = estimate_co2("taxi", 10.0, occupancy=4)
        assert shared.co2_grams == pytest.approx(solo.co2_grams / 4.0)

    def test_occupancy_floors_at_one(self):
        """A zero or fractional occupancy must not divide by zero."""
        assert estimate_co2("taxi", 10.0, occupancy=0).co2_grams == pytest.approx(
            estimate_co2("taxi", 10.0, occupancy=1).co2_grams)

    def test_taxi_dirtier_than_bike_for_same_trip(self):
        assert estimate_co2("taxi", 5.0).co2_grams > estimate_co2("bike", 5.0).co2_grams

    def test_kg_property(self):
        result = estimate_co2("taxi", 10.0)
        assert result.co2_kg == pytest.approx(result.co2_grams / 1000.0)


class TestHumanise:
    def test_negligible_for_tiny_amounts(self):
        assert humanise_co2(0.4) == "negligible"

    def test_phone_charges_for_small_amounts(self):
        assert "smartphone" in humanise_co2(100.0)

    def test_tree_days_for_large_amounts(self):
        assert "tree" in humanise_co2(3000.0)

    def test_handles_negative(self):
        """A negative saving is still describable in magnitude."""
        assert humanise_co2(-3000.0) == humanise_co2(3000.0)


class TestPublishedClaims:
    """Ratios quoted in the README and the app must stay true to the config.

    These numbers appear in prose, where nothing would otherwise catch it if a
    factor were retuned and the claim silently became false.
    """

    def test_operational_scooter_advantage_is_about_27x(self):
        taxi = estimate_co2("taxi", 10.0, basis="operational").co2_grams
        scooter = estimate_co2("scooter", 10.0, basis="operational").co2_grams
        assert taxi / scooter == pytest.approx(27.3, abs=0.5)

    def test_lifecycle_scooter_advantage_is_about_6x(self):
        taxi = estimate_co2("taxi", 10.0, basis="lifecycle").co2_grams
        scooter = estimate_co2("scooter", 10.0, basis="lifecycle").co2_grams
        assert taxi / scooter == pytest.approx(6.3, abs=0.3)

    def test_accounting_basis_changes_the_headline_materially(self):
        """The point of the README passage: the basis is not a detail."""
        operational = (estimate_co2("taxi", 10.0, basis="operational").co2_grams
                       / estimate_co2("scooter", 10.0, basis="operational").co2_grams)
        lifecycle = (estimate_co2("taxi", 10.0, basis="lifecycle").co2_grams
                     / estimate_co2("scooter", 10.0, basis="lifecycle").co2_grams)
        assert operational / lifecycle > 4.0

    def test_lifecycle_taxi_vs_bike_is_about_56x(self):
        taxi = estimate_co2("taxi", 10.0, basis="lifecycle").co2_grams
        bike = estimate_co2("bike", 10.0, basis="lifecycle").co2_grams
        assert taxi / bike == pytest.approx(56.2, abs=1.0)


class TestProvenance:
    def test_every_mode_cites_a_source(self):
        rows = factor_provenance()
        assert {r["mode"] for r in rows} >= {"taxi", "bike", "scooter"}
        for row in rows:
            assert row["source"].strip(), f"{row['mode']} has no citation"
            assert row["g_per_km"] >= 0
