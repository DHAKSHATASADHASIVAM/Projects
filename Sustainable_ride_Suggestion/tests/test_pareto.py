"""Pareto dominance and weighted scoring.

The edge cases here matter more than the happy path: degenerate ties and
floating-point dust are exactly what turn a ranking non-deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from sustainable_ride.recommender.pareto import (
    dominates,
    normalise,
    pareto_frontier,
    weighted_scores,
)


class TestDominates:
    def test_strictly_better_on_all(self):
        assert dominates([1, 1, 1], [2, 2, 2])

    def test_better_on_one_equal_on_rest(self):
        assert dominates([1, 2, 2], [2, 2, 2])

    def test_identical_does_not_dominate(self):
        """Dominance is strict -- a tie is not a win."""
        assert not dominates([1, 1, 1], [1, 1, 1])

    def test_mixed_does_not_dominate(self):
        assert not dominates([1, 3], [2, 2])
        assert not dominates([2, 2], [1, 3])

    def test_worse_does_not_dominate(self):
        assert not dominates([3, 3], [1, 1])

    def test_floating_point_dust_is_not_dominance(self):
        """A 1e-15 difference must not count as a real advantage."""
        assert not dominates([1.0, 1.0], [1.0 + 1e-15, 1.0])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="differ in length"):
            dominates([1, 2], [1, 2, 3])


class TestParetoFrontier:
    def test_all_non_dominated(self):
        # Classic trade-off: each option wins on exactly one objective.
        objectives = [[1, 3], [2, 2], [3, 1]]
        assert pareto_frontier(objectives) == [0, 1, 2]

    def test_dominated_option_excluded(self):
        objectives = [[1, 1], [2, 2], [3, 1]]
        frontier = pareto_frontier(objectives)
        assert 0 in frontier
        assert 1 not in frontier      # dominated by [1, 1]

    def test_single_option(self):
        assert pareto_frontier([[5, 5, 5]]) == [0]

    def test_identical_options_all_survive(self):
        """Ties dominate nobody, so duplicates all stay on the frontier."""
        assert pareto_frontier([[1, 1], [1, 1], [1, 1]]) == [0, 1, 2]

    def test_realistic_three_mode_case(self):
        # cost, time, co2 -- taxi fast+dirty+dear, bike slow+clean+cheap
        taxi = [35.0, 30.0, 3400.0]
        bike = [7.4, 42.0, 55.0]
        scooter = [12.0, 31.0, 490.0]
        frontier = pareto_frontier([taxi, bike, scooter])
        assert set(frontier) == {0, 1, 2}

    def test_strictly_worse_mode_is_dropped(self):
        good = [5.0, 10.0, 100.0]
        worse = [6.0, 11.0, 110.0]        # worse on every axis
        assert pareto_frontier([good, worse]) == [0]

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            pareto_frontier([1, 2, 3])


class TestNormalise:
    def test_maps_to_unit_range(self):
        result = normalise([10.0, 20.0, 30.0])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_all_equal_returns_zeros(self):
        """A dimension nothing differs on must contribute nothing, not divide by zero."""
        result = normalise([5.0, 5.0, 5.0])
        assert np.allclose(result, 0.0)

    def test_single_value(self):
        assert np.allclose(normalise([42.0]), 0.0)

    def test_empty(self):
        assert normalise([]).size == 0


class TestWeightedScores:
    def test_lower_is_better(self):
        scores = weighted_scores(cost=[1.0, 10.0], time=[1.0, 10.0],
                                 co2=[1.0, 10.0], weights={"cost": 1, "time": 1, "co2": 1})
        assert scores[0] < scores[1]

    def test_weights_are_renormalised(self):
        """Unnormalised weights must give the same ranking as normalised ones."""
        cost, time, co2 = [1.0, 5.0], [5.0, 1.0], [3.0, 3.0]
        a = weighted_scores(cost, time, co2, {"cost": 2, "time": 1, "co2": 1})
        b = weighted_scores(cost, time, co2, {"cost": 0.5, "time": 0.25, "co2": 0.25})
        assert np.allclose(a, b)

    def test_single_objective_weighting(self):
        """With all weight on CO2, the cleanest option must win outright."""
        scores = weighted_scores(cost=[1.0, 100.0], time=[1.0, 100.0],
                                 co2=[100.0, 1.0],
                                 weights={"cost": 0, "time": 0, "co2": 1})
        assert scores[1] < scores[0]

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError, match="must be positive"):
            weighted_scores([1.0], [1.0], [1.0], {"cost": 0, "time": 0, "co2": 0})

    def test_tied_dimension_does_not_break_scoring(self):
        scores = weighted_scores(cost=[5.0, 5.0], time=[1.0, 9.0], co2=[5.0, 5.0],
                                 weights={"cost": 1, "time": 1, "co2": 1})
        assert np.all(np.isfinite(scores))
        assert scores[0] < scores[1]
