import json

import numpy as np
import pandas as pd

from app.data_pipeline import build_delay_dataset
from scripts._provenance import safe_source_url
from scripts.generate_synthetic_data import generate


def test_synthetic_generator_repeats_same_records(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    generate(first, days=3, manifest_path=first_manifest)
    generate(second, days=3, manifest_path=second_manifest)
    assert first.read_bytes() == second.read_bytes()
    first_data = json.loads(first_manifest.read_text(encoding="utf-8"))
    second_data = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert first_data["sha256"] == second_data["sha256"]
    assert first_data["seed"] == second_data["seed"] == 42


def test_synthetic_generator_has_persistent_activity_and_variable_dwell(tmp_path):
    output = tmp_path / "development.csv"
    manifest = tmp_path / "provenance.json"
    generate(output, days=180, manifest_path=manifest)
    frame = pd.read_csv(output, parse_dates=["BaseDateTime"])
    frame["BaseDateTime"] = pd.to_datetime(
        frame["BaseDateTime"], utc=True, format="mixed"
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))

    daily_vessels = frame.groupby(frame["BaseDateTime"].dt.floor("D"))["MMSI"].nunique()
    indices = np.arange(len(daily_vessels))
    expected = (
        48
        + 8 * np.sin(2 * np.pi * indices / 365.25)
        + 5 * np.sin(2 * np.pi * indices / 7)
    )
    residuals = daily_vessels.to_numpy() - expected
    lag_one_correlation = np.corrcoef(residuals[1:], residuals[:-1])[0, 1]
    assert lag_one_correlation > 0.45

    anchored = frame.loc[
        frame["LAT"].between(33.68, 33.82)
        & frame["LON"].between(-118.49, -118.31)
        & frame["SOG"].between(0, 0.5)
    ].copy()
    anchored["date"] = anchored["BaseDateTime"].dt.floor("D")
    spans = anchored.groupby(["date", "MMSI"])["BaseDateTime"].agg(
        lambda timestamps: (timestamps.max() - timestamps.min()).total_seconds() / 3600
    )
    assert spans.nunique() > 1
    assert manifest_data["synthetic_process"]["latent_noise_persistence"] == 0.8
    assert "not ground-truth waiting time" in manifest_data["synthetic_process"][
        "anchorage_dwell_hours"
    ]


def test_synthetic_generator_supports_anchorage_dwell_and_delay_analytics(tmp_path):
    output = tmp_path / "development.csv"
    manifest = tmp_path / "provenance.json"
    generate(output, days=90, manifest_path=manifest)
    raw = pd.read_csv(output)
    processed, quality = build_delay_dataset(raw)

    assert quality["original_columns"] == list(raw.columns)
    assert quality["candidate_anchorage_days"] > 0
    assert quality["dwell_measurement_days"] > 0
    assert (processed["anchorage_vessels"] == 0).any()
    assert processed["anchorage_vessels"].nunique() > 5
    assert processed["anchorage_dwell_hours"].dropna().nunique() > 5
    assert quality["congestion_label_distribution"]["not_congested"] > 0
    assert quality["congestion_label_distribution"]["congested"] > 0
    assert quality["delay_target_distribution"]["not_delayed"] > 0
    assert quality["delay_target_distribution"]["delayed"] > 0
    assert "SyntheticCongestionLabel" not in processed.columns
    assert "SyntheticDelayMinutes" not in processed.columns
    assert "SyntheticDelayLabel" not in processed.columns
    assert json.loads(manifest.read_text(encoding="utf-8"))["data_status"] == "synthetic"


def test_source_url_manifest_redacts_query_and_credentials():
    sanitized, redacted = safe_source_url(
        "https://user:password@example.org/path.csv?token=sensitive#fragment"
    )
    assert sanitized == "https://example.org/path.csv"
    assert redacted
