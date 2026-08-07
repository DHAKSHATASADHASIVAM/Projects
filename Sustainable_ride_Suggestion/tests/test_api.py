"""HTTP API contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sustainable_ride.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        if not test_client.get("/health").json()["models_loaded"]:
            pytest.skip("Models not trained; run `python -m sustainable_ride.cli pipeline`")
        yield test_client


VALID_REQUEST = {
    "origin": {"lat": 40.7580, "lon": -73.9855},
    "destination": {"lat": 40.7061, "lon": -73.9969},
    "departure": "2024-01-15T17:30:00",
    "weights": {"cost": 0.2, "time": 0.3, "co2": 0.5},
    "include_geometry": False,
}


class TestMetaRoutes:
    def test_health(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["models_loaded"] is True

    def test_models_route_exposes_metrics(self, client):
        payload = client.get("/models").json()
        assert set(payload["models"]) >= {"taxi_duration", "taxi_fare", "bike_duration"}
        for spec in payload["models"].values():
            assert "metrics" in spec
            assert spec["metrics"]["mae"] > 0

    def test_emission_factors_are_cited(self, client):
        payload = client.get("/emissions/factors").json()
        for factor in payload["factors"]:
            assert factor["source"].strip()


class TestRecommend:
    def test_happy_path(self, client):
        response = client.post("/recommend", json=VALID_REQUEST)
        assert response.status_code == 200
        payload = response.json()
        assert payload["best_mode"] in {"taxi", "bike", "scooter"}
        assert len(payload["options"]) == 3
        assert payload["narrative"]

    def test_options_are_ranked(self, client):
        payload = client.post("/recommend", json=VALID_REQUEST).json()
        ranks = [o["rank"] for o in payload["options"] if o["rank"] is not None]
        assert ranks == sorted(ranks)

    def test_defaults_applied_when_fields_omitted(self, client):
        response = client.post("/recommend", json={
            "origin": {"lat": 40.7580, "lon": -73.9855},
            "destination": {"lat": 40.7061, "lon": -73.9969},
        })
        assert response.status_code == 200

    def test_geometry_suppressed_when_not_requested(self, client):
        payload = client.post("/recommend", json=VALID_REQUEST).json()
        assert all(o["geometry"] == [] for o in payload["options"])


class TestValidation:
    def test_rejects_coordinates_outside_nyc(self, client):
        response = client.post("/recommend", json={
            **VALID_REQUEST, "origin": {"lat": 51.5074, "lon": -0.1278}})
        assert response.status_code == 422

    def test_rejects_identical_endpoints(self, client):
        point = {"lat": 40.7580, "lon": -73.9855}
        response = client.post("/recommend", json={
            **VALID_REQUEST, "origin": point, "destination": point})
        assert response.status_code == 422

    def test_rejects_all_zero_weights(self, client):
        response = client.post("/recommend", json={
            **VALID_REQUEST, "weights": {"cost": 0, "time": 0, "co2": 0}})
        assert response.status_code == 422

    def test_rejects_out_of_range_passenger_count(self, client):
        response = client.post("/recommend", json={
            **VALID_REQUEST, "passenger_count": 99})
        assert response.status_code == 422

    def test_rejects_out_of_range_rain_probability(self, client):
        response = client.post("/recommend", json={
            **VALID_REQUEST, "rain_probability": 3.0})
        assert response.status_code == 422

    def test_rejects_missing_destination(self, client):
        response = client.post("/recommend", json={
            "origin": {"lat": 40.7580, "lon": -73.9855}})
        assert response.status_code == 422
