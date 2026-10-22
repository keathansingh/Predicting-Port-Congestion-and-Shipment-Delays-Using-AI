import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import delay_training
from app.main import app


def labeled_daily_data(size=180):
    dates = pd.date_range("2023-01-01", periods=size, freq="D", tz="UTC")
    index = np.arange(size)
    target = ((index // 4) % 2).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "shipment_delay": target,
            "vessel_arrivals_daily": 20 + index % 25,
            "queue_length_vessels": 5 + index % 30,
            "port_capacity_berths": 18 + index % 4,
            "weather_condition": np.where(index % 3, "Clear", "Rain"),
            "wind_speed_kmh": 6 + index % 35,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
        }
    )


def test_delay_training_uses_temporal_holdout_and_writes_distinct_artifacts(
    monkeypatch, tmp_path
):
    processed = tmp_path / "port_daily.csv"
    provenance_path = tmp_path / "provenance.json"
    model_path = tmp_path / "delay_model.joblib"
    metrics_path = tmp_path / "delay_evaluation.json"
    labeled_daily_data().to_csv(processed, index=False)
    provenance_path.write_text(
        json.dumps(
            {
                "data_status": "uploaded",
                "source_name": "synthetic training fixture",
                "filename": "fixture.csv",
                "sha256": "delay-fixture-hash",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(delay_training, "PROCESSED_PATH", processed)
    monkeypatch.setattr(delay_training, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(delay_training, "DELAY_MODEL_PATH", model_path)
    monkeypatch.setattr(delay_training, "DELAY_METRICS_PATH", metrics_path)

    bundle = delay_training.load_delay_model()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert bundle["status"] == "available"
    assert bundle["training_dataset"]["data_status"] == "synthetic"
    assert "shipment_delay" not in bundle["features"]
    assert "SyntheticDelayMinutes" not in bundle["features"]
    assert metrics["test"]["sample_count"] == 36
    assert metrics["test"]["period"]["start"] > metrics["train"]["period"]["end"]
    assert metrics["test"]["precision"] is not None
    assert metrics["test"]["recall"] is not None
    assert metrics["test"]["f1"] is not None
    assert metrics["test"]["roc_auc"] is not None
    confusion = metrics["test"]["confusion_matrix"]
    assert len(confusion) == 2 and all(len(row) == 2 for row in confusion)
    assert sum(sum(row) for row in confusion) == metrics["test"]["sample_count"]
    assert metrics["probability_calibration"] == "uncalibrated"
    assert metrics["dataset_identity"]["sha256"] == "delay-fixture-hash"
    assert model_path.is_file()


def test_delay_training_returns_unavailable_for_single_class_data():
    frame = labeled_daily_data(90)
    frame["shipment_delay"] = 1

    bundle, metrics = delay_training.train_delay_model(frame, {})

    assert bundle["status"] == "unavailable"
    assert "both classes" in bundle["reason"]
    assert metrics["available"] is False


def test_missing_delay_labels_and_unreadable_artifact_return_unavailable(
    monkeypatch, tmp_path
):
    processed = tmp_path / "port_daily.csv"
    provenance_path = tmp_path / "provenance.json"
    model_path = tmp_path / "delay_model.joblib"
    metrics_path = tmp_path / "delay_evaluation.json"
    labeled_daily_data().drop(columns=["shipment_delay"]).to_csv(processed, index=False)
    provenance_path.write_text("{}", encoding="utf-8")
    model_path.write_bytes(b"not a joblib artifact")
    metrics_path.write_text('{"previous":"evaluation"}', encoding="utf-8")
    monkeypatch.setattr(delay_training, "PROCESSED_PATH", processed)
    monkeypatch.setattr(delay_training, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(delay_training, "DELAY_MODEL_PATH", model_path)
    monkeypatch.setattr(delay_training, "DELAY_METRICS_PATH", metrics_path)

    bundle = delay_training.load_delay_model()

    assert bundle["status"] == "unavailable"
    assert "SyntheticDelayLabel" in bundle["reason"]
    assert metrics_path.read_text(encoding="utf-8") == '{"previous":"evaluation"}'


def test_vercel_does_not_train_or_overwrite_artifacts_for_a_mismatched_dataset(
    monkeypatch, tmp_path
):
    processed = tmp_path / "port_daily.csv"
    provenance_path = tmp_path / "provenance.json"
    model_path = tmp_path / "delay_model.joblib"
    metrics_path = tmp_path / "delay_evaluation.json"
    labeled_daily_data().to_csv(processed, index=False)
    provenance_path.write_text(
        json.dumps(
            {
                "data_status": "synthetic",
                "source_name": "synthetic Vercel fixture",
                "sha256": "active-dataset-hash",
            }
        ),
        encoding="utf-8",
    )
    delay_training.joblib.dump(
        {
            "artifact_version": delay_training.DELAY_ARTIFACT_VERSION,
            "training_dataset": {"sha256": "bundled-dataset-hash"},
        },
        model_path,
    )
    metrics_path.write_text('{"previous":"evaluation"}', encoding="utf-8")
    original_model = model_path.read_bytes()
    monkeypatch.setattr(delay_training, "PROCESSED_PATH", processed)
    monkeypatch.setattr(delay_training, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(delay_training, "DELAY_MODEL_PATH", model_path)
    monkeypatch.setattr(delay_training, "DELAY_METRICS_PATH", metrics_path)
    monkeypatch.setenv("VERCEL", "1")

    bundle = delay_training.load_delay_model()

    assert bundle["status"] == "unavailable"
    assert "differs from the bundled" in bundle["reason"]
    assert model_path.read_bytes() == original_model
    assert metrics_path.read_text(encoding="utf-8") == '{"previous":"evaluation"}'


def test_delay_prediction_endpoint_uses_model_and_has_explicit_unavailable_state():
    bundle, _ = delay_training.train_delay_model(
        labeled_daily_data(),
        {
            "data_status": "synthetic",
            "source_name": "synthetic endpoint fixture",
            "sha256": "endpoint-fixture-hash",
        },
    )
    request_body = {
        "vessel_arrivals_daily": 35,
        "queue_length_vessels": 27,
        "port_capacity_berths": 20,
        "weather_condition": "Rain",
        "wind_speed_kmh": 22,
        "day_of_week": 2,
        "month": 8,
    }
    with TestClient(app) as client:
        loaded = app.state.delay_model_bundle
        app.state.delay_model_bundle = bundle
        try:
            prediction = client.post("/api/v1/delay/predict", json=request_body)
            unavailable_bundle = {"status": "unavailable", "reason": "Missing binary labels."}
            app.state.delay_model_bundle = unavailable_bundle
            unavailable = client.post("/api/v1/delay/predict", json=request_body)
        finally:
            app.state.delay_model_bundle = loaded

    assert prediction.status_code == 200, prediction.text
    body = prediction.json()
    assert body["status"] == body["model_availability"] == "available"
    assert body["predicted_outcome"] in {"delayed", "not_delayed"}
    assert 0 <= body["delay_probability_percent"] <= 100
    assert body["risk_category"] in {"Low", "Medium", "High"}
    assert body["estimated_delay_minutes"] is None
    assert body["probability_calibration"] == "uncalibrated"
    assert body["dataset_provenance"]["sha256"] == "endpoint-fixture-hash"
    assert body["main_factors"]
    assert body["recommendations"]
    assert unavailable.status_code == 200
    assert unavailable.json()["status"] == "unavailable"
    assert unavailable.json()["delay_probability_percent"] is None
    assert "Missing binary labels" in unavailable.json()["reason"]
