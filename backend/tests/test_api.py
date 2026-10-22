from fastapi.testclient import TestClient

from app.main import app
from app.config import FEATURES


def test_health_reports_artifact_state():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert {"status", "model_loaded", "data_available"} <= response.json().keys()


def test_metadata_and_metrics_do_not_fabricate_results():
    with TestClient(app) as client:
        metadata = client.get("/api/v1/metadata")
        metrics = client.get("/api/v1/metrics")
    assert metadata.status_code == metrics.status_code == 200
    assert metadata.json()["target"] == "next_day_congestion"
    assert isinstance(metrics.json()["available"], bool)
    assert metadata.headers["cache-control"] == "no-store"


def test_invalid_prediction_request_is_rejected():
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"vessel_density": -1})
    assert response.status_code == 422


def test_prediction_rejects_anchorage_count_above_vessel_density():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json={
                "vessel_density": 2,
                "anchorage_vessels": 3,
                "anchorage_dwell_hours": 12,
                "day_of_week": 2,
                "month": 8,
            },
        )
    assert response.status_code == 422
    assert "anchorage_vessels cannot exceed vessel_density" in response.text


def test_untrained_model_fails_with_service_unavailable():
    with TestClient(app) as client:
        loaded = app.state.model_bundle
        app.state.model_bundle = None
        try:
            response = client.post(
                "/api/v1/predict",
                json={
                    "vessel_density": 42,
                    "anchorage_vessels": 18,
                    "anchorage_dwell_hours": 12,
                    "day_of_week": 2,
                    "month": 8,
                },
            )
        finally:
            app.state.model_bundle = loaded
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_cors_allows_local_frontend_origins():
    with TestClient(app) as client:
        for origin in (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ):
            response = client.options(
                "/api/v1/metadata",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin


def test_prediction_has_separate_reliability_and_explanations():
    class FixedModel:
        def predict_proba(self, frame):
            import numpy as np

            assert list(frame.columns) == FEATURES
            return np.asarray([[0.25, 0.75]])

    with TestClient(app) as client:
        loaded = app.state.model_bundle
        app.state.model_bundle = {
            "model": FixedModel(),
            "features": FEATURES,
            "threshold": 0.5,
            "reference_values": {
                "vessel_density": 30,
                "anchorage_vessels": 10,
                "anchorage_dwell_hours": 4,
                "day_of_week": 2,
                "month": 6,
            },
            "feature_ranges": {
                "vessel_density": [10, 80],
                "anchorage_vessels": [2, 40],
                "anchorage_dwell_hours": [0, 24],
                "day_of_week": [0, 6],
                "month": [1, 12],
            },
            "training_samples": 100,
            "calibration_samples": 50,
            "calibration_brier": 0.1,
            "heldout_test_brier": 0.1,
            "heldout_test_brier_skill": 0.2,
            "heldout_test_roc_auc": 0.8,
            "heldout_test_samples": 40,
            "heldout_test_has_both_classes": True,
            "calibration_status": "sigmoid_calibrated_on_validation",
            "data_status": "public",
            "model_version": "test-v1",
            "target_definition": "Unit-test target definition",
            "target_status": "provided",
            "target_source": "SyntheticCongestionLabel fixture",
        }
        try:
            response = client.post(
                "/api/v1/predict",
                json={
                    "vessel_density": 42,
                    "anchorage_vessels": 18,
                    "anchorage_dwell_hours": 12,
                    "day_of_week": 2,
                    "month": 8,
                },
            )
            out_of_range_response = client.post(
                "/api/v1/predict",
                json={
                    "vessel_density": 90,
                    "anchorage_vessels": 18,
                    "anchorage_dwell_hours": 12,
                    "day_of_week": 2,
                    "month": 8,
                },
            )
            app.state.model_bundle["data_status"] = "synthetic"
            synthetic_response = client.post(
                "/api/v1/predict",
                json={
                    "vessel_density": 42,
                    "anchorage_vessels": 18,
                    "anchorage_dwell_hours": 12,
                    "day_of_week": 2,
                    "month": 8,
                },
            )
        finally:
            app.state.model_bundle = loaded
    assert response.status_code == 200
    body = response.json()
    assert body["congestion_probability"] == 0.75
    assert body["prediction_reliability"] == "high"
    assert body["target_status"] == "provided"
    assert body["target_source"] == "SyntheticCongestionLabel fixture"
    assert any("supplied congestion label" in warning for warning in body["data_quality_warnings"])
    assert len(body["local_explanations"]) == 5
    assert body["data_quality_warnings"]
    assert out_of_range_response.status_code == 200
    out_of_range_body = out_of_range_response.json()
    assert out_of_range_body["prediction_reliability"] == "low"
    assert any("outside the observed training range" in warning for warning in out_of_range_body["data_quality_warnings"])
    assert synthetic_response.status_code == 200
    synthetic_body = synthetic_response.json()
    assert synthetic_body["prediction_reliability"] == "low"
    assert any("synthetic development data" in reason for reason in synthetic_body["reliability_reasons"])


def test_prediction_rejects_model_feature_schema_mismatch():
    with TestClient(app) as client:
        loaded = app.state.model_bundle
        app.state.model_bundle = {
            "model": object(),
            "features": ["different_feature"],
        }
        try:
            response = client.post(
                "/api/v1/predict",
                json={
                    "vessel_density": 42,
                    "anchorage_vessels": 18,
                    "anchorage_dwell_hours": 12,
                    "day_of_week": 2,
                    "month": 8,
                },
            )
        finally:
            app.state.model_bundle = loaded
    assert response.status_code == 503
    assert "feature schema is incompatible" in response.json()["detail"]
