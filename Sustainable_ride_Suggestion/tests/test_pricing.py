"""Fare calculation."""

from __future__ import annotations

import pytest

from sustainable_ride.pricing import (
    FareBreakdown,
    bike_fare,
    scooter_fare,
    taxi_fare_analytic,
)

MIDDAY, EVENING_PEAK, LATE_NIGHT = 12, 17, 23


class TestTaxiFare:
    def test_includes_initial_charge(self):
        fare = taxi_fare_analytic(1.0, 5.0, MIDDAY, is_weekday=True)
        assert fare.base_usd == pytest.approx(3.00)

    def test_monotonic_in_distance(self):
        short = taxi_fare_analytic(2.0, 8.0, MIDDAY, True).total_usd
        long = taxi_fare_analytic(10.0, 25.0, MIDDAY, True).total_usd
        assert long > short

    def test_night_surcharge_applied(self):
        day = taxi_fare_analytic(5.0, 15.0, MIDDAY, True)
        night = taxi_fare_analytic(5.0, 15.0, LATE_NIGHT, True)
        assert "night" in night.surcharges
        assert "night" not in day.surcharges
        assert night.total_usd > day.total_usd

    def test_weekday_peak_surcharge(self):
        weekday = taxi_fare_analytic(5.0, 15.0, EVENING_PEAK, is_weekday=True)
        weekend = taxi_fare_analytic(5.0, 15.0, EVENING_PEAK, is_weekday=False)
        assert "weekday_peak" in weekday.surcharges
        assert "weekday_peak" not in weekend.surcharges

    def test_standard_surcharges_always_present(self):
        fare = taxi_fare_analytic(3.0, 10.0, MIDDAY, True)
        assert "mta_state" in fare.surcharges
        assert "improvement" in fare.surcharges

    def test_congestion_zone_detected_in_manhattan(self):
        fare = taxi_fare_analytic(3.0, 10.0, MIDDAY, True,
                                  origin=(40.7580, -73.9855),
                                  destination=(40.7061, -73.9969))
        assert "congestion" in fare.surcharges

    def test_congestion_zone_absent_outside_manhattan(self):
        fare = taxi_fare_analytic(3.0, 10.0, MIDDAY, True,
                                  origin=(40.6602, -73.9690),   # Prospect Park
                                  destination=(40.6782, -73.9442))
        assert "congestion" not in fare.surcharges

    def test_slow_traffic_charges_time(self):
        """Below 12 mph the meter bills time, so a crawl costs more than a cruise."""
        fast = taxi_fare_analytic(5.0, 10.0, MIDDAY, True)   # ~30 km/h
        slow = taxi_fare_analytic(5.0, 45.0, MIDDAY, True)   # ~6.7 km/h
        assert slow.time_usd > 0
        assert fast.time_usd == 0
        assert slow.total_usd > fast.total_usd

    def test_tip_excluded_by_default(self):
        assert taxi_fare_analytic(5.0, 15.0, MIDDAY, True).tip_usd == 0.0

    def test_tip_included_when_requested(self):
        fare = taxi_fare_analytic(5.0, 15.0, MIDDAY, True, include_tip=True)
        assert fare.tip_usd > 0
        assert fare.total_usd > taxi_fare_analytic(5.0, 15.0, MIDDAY, True).total_usd

    def test_zero_distance_still_charges_base(self):
        fare = taxi_fare_analytic(0.0, 0.0, MIDDAY, True)
        assert fare.total_usd >= 3.00

    def test_negative_inputs_clamped(self):
        fare = taxi_fare_analytic(-5.0, -10.0, MIDDAY, True)
        assert fare.total_usd >= 3.00

    def test_surcharge_total_matches_sum(self):
        fare = taxi_fare_analytic(5.0, 15.0, LATE_NIGHT, True)
        assert fare.surcharge_total == pytest.approx(sum(fare.surcharges.values()))


class TestBikeFare:
    def test_short_ride_is_unlock_only(self):
        """Citi Bike's classic single ride includes the first 30 minutes."""
        fare = bike_fare(15.0, electric=False)
        assert fare.time_usd == 0.0
        assert fare.total_usd == pytest.approx(fare.base_usd)

    def test_overage_charged_beyond_included_minutes(self):
        short = bike_fare(20.0, electric=False).total_usd
        long = bike_fare(50.0, electric=False).total_usd
        assert long > short

    def test_electric_charges_from_the_first_minute(self):
        assert bike_fare(10.0, electric=True).time_usd > 0

    def test_electric_costs_more_than_classic(self):
        assert bike_fare(10.0, electric=True).total_usd > \
               bike_fare(10.0, electric=False).total_usd

    def test_member_plan_is_cheaper(self):
        single = bike_fare(20.0, electric=False, plan="single_ride").total_usd
        member = bike_fare(20.0, electric=False, plan="member").total_usd
        assert member < single

    def test_monotonic_in_duration(self):
        durations = [5, 20, 35, 60, 90]
        totals = [bike_fare(d).total_usd for d in durations]
        assert totals == sorted(totals)

    def test_negative_duration_clamped(self):
        assert bike_fare(-10.0).total_usd >= 0


class TestScooterFare:
    def test_unlock_plus_time(self):
        fare = scooter_fare(10.0)
        assert fare.base_usd > 0
        assert fare.time_usd > 0

    def test_monotonic_in_duration(self):
        assert scooter_fare(20.0).total_usd > scooter_fare(5.0).total_usd

    def test_minimum_fare_enforced(self):
        assert scooter_fare(0.0).total_usd >= 1.50

    def test_negative_duration_clamped(self):
        assert scooter_fare(-5.0).total_usd >= 0


class TestFareBreakdown:
    def test_is_frozen(self):
        fare = FareBreakdown(mode="taxi", total_usd=10.0)
        with pytest.raises(Exception):
            fare.total_usd = 20.0

    def test_carries_a_note(self):
        """Every estimate should explain its own caveats."""
        for fare in (taxi_fare_analytic(5.0, 15.0, MIDDAY, True),
                     bike_fare(10.0), scooter_fare(10.0)):
            assert fare.notes.strip()
