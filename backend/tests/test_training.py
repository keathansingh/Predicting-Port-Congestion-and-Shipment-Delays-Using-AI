import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app import model_training
from app.config import FEATURES


def make_daily_data(size=150):
    dates = pd.date_range("2022-01-01", periods=size, freq="D", tz="UTC")
    vessel_density = 30 + (np.arange(size) % 31)
    anchored = (np.arange(size) % 20) + 4
    return pd.DataFrame(
        {
            "date": dates,
            "vessel_density": vessel_density,
            "anchorage_vessels": anchored,
            "anchorage_dwell_hours": (np.arange(size) % 15).astype(float),
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "next_day_congestion": (anchored >= 14).astype(int),
        }
    )


def test_training_splits_chronologically_and_serializes(monkeypatch, tmp_path):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    processed = data_dir / "port_daily.csv"
    make_daily_data().to_csv(processed, index=False)
    (data_dir / "data_quality.json").write_text(
        json.dumps(
            {
                "processed_rows": 150,
                "target_status": "provided",
                "target_definition": "Supplied fixture congestion labels.",
                "target_source": "SyntheticCongestionLabel fixture",
            }
        ),
        encoding="utf-8",
    )
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir()
    (manifests / "provenance.json").write_text(
        json.dumps(
            {
                "data_status": "synthetic",
                "source_name": "training fixture",
                "filename": "fixture.csv",
                "sha256": "fixture-hash",
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "artifacts" / "models" / "model.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics" / "metrics.json"
    plots_dir = tmp_path / "artifacts" / "metrics"
    monkeypatch.setattr(model_training, "PROCESSED_PATH", processed)
    monkeypatch.setattr(model_training, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_training, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(model_training, "PLOTS_DIR", plots_dir)

    result = model_training.train_and_evaluate()

    assert result["splits"]["train"]["period"]["end"] < result["splits"]["calibration"]["period"]["start"]
    assert result["splits"]["calibration"]["period"]["end"] < result["splits"]["validation"]["period"]["start"]
    assert result["splits"]["validation"]["period"]["end"] < result["splits"]["test"]["period"]["start"]
    assert result["test"]["sample_count"] == 16
    assert result["threshold_selection"]["partition"] == "validation"
    assert result["threshold_selection"]["sample_count"] == 22
    assert sum(result["class_distribution"]["calibration"].values()) == 22
    assert result["test"]["selected_model"]["classification_threshold"] == result[
        "threshold_selection"
    ]["selected_threshold"]
    assert sum(sum(row) for row in result["test"]["selected_model"]["confusion_matrix"]) == 16
    folds = result["training_only_temporal_cv"]["logistic_regression"]["folds"]
    assert len(folds) == 5
    assert all(fold["training_end"] < fold["validation_start"] for fold in folds)
    assert result["data_status"] == "synthetic"
    assert result["test"]["selected_model"]["accuracy"] is not None
    assert sum(result["test"]["risk_distribution"].values()) == result["test"]["sample_count"]
    assert result["calibration_curve"]["points"]
    assert not result["test"]["selected_model_is_calibrated"]
    importance = list(result["global_feature_importance"].values())
    assert importance == sorted(importance, reverse=True)
    bundle = joblib.load(model_path)
    probabilities = bundle["model"].predict_proba(make_daily_data().loc[134:, FEATURES])[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert bundle["model_version"] == result["model_version"]
    assert bundle["features"] == FEATURES
    assert bundle["training_dataset"]["sha256"] == "fixture-hash"
    assert result["target_status"] == bundle["target_status"] == "provided"
    assert result["target_definition"] == bundle["target_definition"] == "Supplied fixture congestion labels."
    assert result["target_source"] == bundle["target_source"] == "SyntheticCongestionLabel fixture"
    assert result["training_dataset"]["source_name"] == "training fixture"
    assert bundle["threshold"] == result["threshold_selection"]["selected_threshold"]
    assert (plots_dir / "calibration.png").is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["test"]["sample_count"] == 16
    altered_test = make_daily_data()
    altered_test.loc[134:, "next_day_congestion"] = (
        1 - altered_test.loc[134:, "next_day_congestion"]
    )
    altered_test.to_csv(processed, index=False)
    with_altered_test_labels = model_training.train_and_evaluate()
    assert (
        with_altered_test_labels["threshold_selection"]["selected_threshold"]
        == result["threshold_selection"]["selected_threshold"]
    )
    make_daily_data().to_csv(processed, index=False)
    repeated = model_training.train_and_evaluate()
    repeated_bundle = joblib.load(model_path)
    repeated_probabilities = repeated_bundle["model"].predict_proba(
        make_daily_data().loc[134:, FEATURES]
    )[:, 1]
    assert repeated["model_version"] == result["model_version"]
    np.testing.assert_allclose(repeated_probabilities, probabilities)


def test_training_rejects_small_and_single_class_training_data(monkeypatch, tmp_path):
    processed = tmp_path / "port_daily.csv"
    monkeypatch.setattr(model_training, "PROCESSED_PATH", processed)
    make_daily_data(size=49).to_csv(processed, index=False)
    with pytest.raises(ValueError, match="At least 50"):
        model_training.train_and_evaluate()
    frame = make_daily_data(size=100)
    frame["next_day_congestion"] = 0
    frame.to_csv(processed, index=False)
    with pytest.raises(ValueError, match="contains one class"):
        model_training.train_and_evaluate()
