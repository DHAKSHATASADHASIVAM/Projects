"""Recommender engine, end to end.

These tests use the real trained models where available, and skip cleanly when
they are absent so that a fresh clone can still run the unit suite.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from sustainable_ride.models.registry import ModelsNotTrainedError
from sustainable_ride.recommender import RideRecommender

TIMES_SQUARE = (40.7580, -73.9855)
WALL_STREET = (40.7061, -73.9969)
JFK = (40.6413, -73.7781)
DEPARTURE = datetime(2024, 1, 15, 17, 30)


@pytest.fixture(scope="module")
def recommender() -> RideRecommender:
    try:
        return RideRecommender()
    except ModelsNotTrainedError:
        pytest.skip("Models not trained; run `python -m sustainable_ride.cli pipeline`")


@pytest.fixture(scope="module")
def result(recommender):
    return recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                 include_geometry=False)


class TestBasicOutput:
    def test_returns_every_mode(self, result):
        assert {o.mode for o in result.options} == {"taxi", "bike", "scooter"}

    def test_picks_a_best_option(self, result):
        assert result.best is not None
        assert result.best.feasible

    def test_estimates_are_positive(self, result):
        for option in result.options:
            assert option.duration_min > 0, option.mode
            assert option.cost_usd > 0, option.mode
            assert option.co2_grams >= 0, option.mode
            assert option.distance_km > 0, option.mode

    def test_door_to_door_exceeds_vehicle_time(self, result):
        """Access and egress are real minutes and must be added on."""
        for option in result.options:
            assert option.duration_min >= option.vehicle_duration_min

    def test_narrative_is_populated(self, result):
        assert len(result.narrative) > 40

    def test_every_estimate_declares_its_source(self, result):
        for option in result.options:
            assert option.duration_source.strip(), option.mode
            assert option.cost_source.strip(), option.mode


class TestEmissionsOrdering:
    def test_taxi_is_the_dirtiest_option(self, result):
        by_mode = {o.mode: o for o in result.options}
        assert by_mode["taxi"].co2_grams > by_mode["scooter"].co2_grams
        assert by_mode["scooter"].co2_grams > by_mode["bike"].co2_grams

    def test_bike_is_cheaper_than_taxi(self, result):
        by_mode = {o.mode: o for o in result.options}
        assert by_mode["bike"].cost_usd < by_mode["taxi"].cost_usd

    def test_savings_are_computed_against_taxi(self, result):
        by_mode = {o.mode: o for o in result.options}
        assert by_mode["bike"].co2_saved_vs_taxi_g > 0
        assert by_mode["taxi"].co2_saved_vs_taxi_g == 0


class TestWeightSensitivity:
    def test_carbon_priority_favours_the_cleanest_mode(self, recommender):
        result = recommender.recommend(
            TIMES_SQUARE, WALL_STREET, DEPARTURE,
            weights={"cost": 0.0, "time": 0.0, "co2": 1.0}, include_geometry=False)
        assert result.best.mode == "bike"

    def test_speed_priority_does_not_pick_the_slowest(self, recommender):
        result = recommender.recommend(
            TIMES_SQUARE, WALL_STREET, DEPARTURE,
            weights={"cost": 0.0, "time": 1.0, "co2": 0.0}, include_geometry=False)
        feasible = [o for o in result.options if o.feasible]
        assert result.best.duration_min == min(o.duration_min for o in feasible)

    def test_cost_priority_picks_the_cheapest(self, recommender):
        result = recommender.recommend(
            TIMES_SQUARE, WALL_STREET, DEPARTURE,
            weights={"cost": 1.0, "time": 0.0, "co2": 0.0}, include_geometry=False)
        feasible = [o for o in result.options if o.feasible]
        assert result.best.cost_usd == min(o.cost_usd for o in feasible)

    def test_weights_change_the_answer(self, recommender):
        """If the weights never mattered, the recommender would be pointless."""
        cheap = recommender.recommend(
            TIMES_SQUARE, WALL_STREET, DEPARTURE,
            weights={"cost": 1.0, "time": 0.0, "co2": 0.0}, include_geometry=False)
        fast = recommender.recommend(
            TIMES_SQUARE, WALL_STREET, DEPARTURE,
            weights={"cost": 0.0, "time": 1.0, "co2": 0.0}, include_geometry=False)
        assert cheap.best.mode != fast.best.mode


class TestFeasibility:
    def test_long_trip_rules_out_micromobility(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, JFK, DEPARTURE,
                                       include_geometry=False)
        by_mode = {o.mode: o for o in result.options}
        assert not by_mode["bike"].feasible
        assert by_mode["bike"].infeasible_reasons
        assert by_mode["taxi"].feasible
        assert result.best.mode == "taxi"

    def test_heavy_rain_rules_out_exposed_modes(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                       rain_probability=0.9, include_geometry=False)
        by_mode = {o.mode: o for o in result.options}
        assert not by_mode["bike"].feasible
        assert not by_mode["scooter"].feasible
        assert result.best.mode == "taxi"

    def test_light_rain_warns_without_excluding(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                       rain_probability=0.4, include_geometry=False)
        bike = next(o for o in result.options if o.mode == "bike")
        assert bike.feasible
        assert any("rain" in w.lower() for w in bike.warnings)

    def test_accessibility_requirement_rules_out_micromobility(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                       accessibility_required=True,
                                       include_geometry=False)
        by_mode = {o.mode: o for o in result.options}
        assert not by_mode["bike"].feasible
        assert not by_mode["scooter"].feasible
        assert result.best.mode == "taxi"

    def test_infeasible_options_are_still_returned(self, recommender):
        """Telling the user what was ruled out, and why, is part of the answer."""
        result = recommender.recommend(TIMES_SQUARE, JFK, DEPARTURE,
                                       include_geometry=False)
        assert len(result.options) == 3

    def test_infeasible_options_are_unscored(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, JFK, DEPARTURE,
                                       include_geometry=False)
        for option in result.options:
            if not option.feasible:
                assert option.score is None


class TestPareto:
    def test_frontier_is_populated(self, result):
        assert result.frontier_modes

    def test_best_option_is_on_the_frontier(self, result):
        """A weighted optimum can never be Pareto-dominated."""
        assert result.best.on_pareto_frontier

    def test_frontier_only_contains_feasible_modes(self, recommender):
        result = recommender.recommend(TIMES_SQUARE, JFK, DEPARTURE,
                                       include_geometry=False)
        infeasible = {o.mode for o in result.options if not o.feasible}
        assert not (set(result.frontier_modes) & infeasible)


class TestValidation:
    def test_rejects_out_of_area_origin(self, recommender):
        with pytest.raises(ValueError, match="outside the NYC service area"):
            recommender.recommend((51.5074, -0.1278), WALL_STREET, DEPARTURE)

    def test_rejects_bad_passenger_count(self, recommender):
        with pytest.raises(ValueError, match="passenger_count"):
            recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                  passenger_count=99)

    def test_rejects_bad_rain_probability(self, recommender):
        with pytest.raises(ValueError, match="rain_probability"):
            recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                  rain_probability=5.0)


class TestOccupancy:
    def test_shared_taxi_splits_emissions(self, recommender):
        solo = recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                     passenger_count=1, include_geometry=False)
        shared = recommender.recommend(TIMES_SQUARE, WALL_STREET, DEPARTURE,
                                       passenger_count=4, include_geometry=False)
        solo_taxi = next(o for o in solo.options if o.mode == "taxi")
        shared_taxi = next(o for o in shared.options if o.mode == "taxi")
        assert shared_taxi.co2_grams < solo_taxi.co2_grams


class TestSerialisation:
    def test_round_trips_to_a_dict(self, result):
        payload = result.to_dict()
        assert payload["best_mode"] == result.best.mode
        assert len(payload["options"]) == len(result.options)
        assert "narrative" in payload
