import json
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import datasets
from app.services import runtime


def ais_csv(
    anchored_vessels: int = 26,
    days: tuple[str, str] = ("2024-01-01", "2024-01-02"),
    vessel_count: int = 30,
) -> bytes:
    rows = []
    for day in days:
        for vessel in range(vessel_count):
            for hour in (6, 18):
                anchored = vessel < anchored_vessels
                rows.append(
                    {
                        "MMSI": f"000000{vessel + 100:03d}",
                        "BaseDateTime": f"{day}T{hour:02}:00:00Z",
                        "LAT": 33.74 if anchored else 33.9,
                        "LON": -118.4 if anchored else -118.2,
                        "SOG": 0.2 if anchored else 5,
                    }
                )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def ais_csv_for_dates(
    days: list[str],
    latitude: float = 40.7,
    longitude: float = -74.0,
    vessel_count: int = 3,
    synthetic_label: int = 1,
) -> bytes:
    rows = []
    for day in days:
        for vessel in range(vessel_count):
            rows.append(
                {
                    "MMSI": f"{vessel + 100:09d}",
                    "BaseDateTime": f"{day}T12:00:00Z",
                    "LAT": latitude,
                    "LON": longitude,
                    "SOG": 5,
                    "SyntheticCongestionLabel": synthetic_label,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def test_upload_reports_unverified_geography_and_does_not_use_synthetic_labels(
    tmp_path, monkeypatch
):
    configure_dataset_paths(tmp_path, monkeypatch)
    contents = ais_csv_for_dates(["2025-01-01", "2025-01-02"])

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("new-york.csv", contents, "text/csv")},
            data={"source_name": "East Coast AIS"},
        )
        snapshot = client.get("/api/v1/dashboard").json()
        ports = client.get("/api/v1/ports").json()["ports"]
        generic_history = client.get(
            "/api/v1/ports/uploaded-ais-area/history"
        )
        los_angeles_history = client.get(
            "/api/v1/ports/port-of-los-angeles/history"
        )

    assert upload.status_code == 200, upload.text
    scope = snapshot["metadata"]["geographic_scope"]
    assert scope["identified"] is False
    assert scope["label"] == "Uploaded AIS dataset — geographic area not verified"
    assert scope["coordinate_coverage"] == {
        "latitude": {"min": 40.7, "max": 40.7},
        "longitude": {"min": -74.0, "max": -74.0},
    }
    assert snapshot["metadata"]["source_name"] == "East Coast AIS"
    assert snapshot["metadata"]["port_name"] == scope["label"]
    assert snapshot["metadata"]["target_status"] == "unavailable"
    assert snapshot["metadata"]["prediction_compatible"] is False
    assert snapshot["summary"]["target_available"] is False
    assert snapshot["summary"]["congestion_days"] is None
    assert snapshot["summary"]["congestion_rate"] is None
    assert all(row["next_day_congestion"] is None for row in snapshot["history"])
    assert all(port["name"] == scope["label"] for port in ports)
    assert generic_history.status_code == 200
    assert los_angeles_history.status_code == 404
    assert snapshot["active_dataset"]["source_name"] == "East Coast AIS"


def test_upload_with_los_angeles_coordinates_reports_geofence_match_not_verified_port(
    tmp_path, monkeypatch
):
    configure_dataset_paths(tmp_path, monkeypatch)
    contents = ais_csv_for_dates(
        ["2025-01-01", "2025-01-02"],
        latitude=33.74,
        longitude=-118.4,
    )

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("coordinate-match.csv", contents, "text/csv")},
        )
        snapshot = client.get("/api/v1/dashboard").json()

    assert upload.status_code == 200, upload.text
    scope = snapshot["metadata"]["geographic_scope"]
    assert scope["identified"] is True
    assert scope["port_id"] == "port-of-los-angeles"
    assert scope["label"] == (
        "Configured Port of Los Angeles development geofence "
        "(coordinate match; port identity not independently verified)"
    )
    assert "does not independently verify port identity" in scope["identification_method"]


def test_legacy_uploaded_geofence_metadata_is_displayed_as_unverified_port_identity(
    tmp_path, monkeypatch
):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={
                "file": (
                    "coordinate-match.csv",
                    ais_csv_for_dates(
                        ["2025-01-01", "2025-01-02"],
                        latitude=33.74,
                        longitude=-118.4,
                    ),
                    "text/csv",
                )
            },
        )
        assert upload.status_code == 200, upload.text
        provenance = json_load(paths["provenance"])
        provenance["geographic_scope"]["label"] = (
            "Port of Los Angeles (development geofence)"
        )
        provenance["geographic_scope"]["identification_method"] = (
            "All uploaded coordinates fall within the configured Port of Los Angeles "
            "development geofence."
        )
        paths["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
        dashboard = client.get("/api/v1/dashboard").json()
        active_dataset = client.get("/api/v1/datasets/active").json()

    expected_label = (
        "Configured Port of Los Angeles development geofence "
        "(coordinate match; port identity not independently verified)"
    )
    assert dashboard["metadata"]["geographic_scope"]["label"] == expected_label
    assert active_dataset["geographic_scope"]["label"] == expected_label
    assert "does not independently verify port identity" in dashboard["metadata"][
        "geographic_scope"
    ]["identification_method"]


def test_date_filtered_dashboard_and_summary_share_selected_period_rows(
    tmp_path, monkeypatch
):
    configure_dataset_paths(tmp_path, monkeypatch)
    dates = pd.date_range("2024-01-01", "2024-08-01", freq="D").strftime("%Y-%m-%d").tolist()
    contents = ais_csv_for_dates(dates, vessel_count=1)

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("long-coverage.csv", contents, "text/csv")},
        )
        assert upload.status_code == 200, upload.text
        one_month = client.get(
            "/api/v1/dashboard",
            params={"start_date": "2024-07-01", "end_date": "2024-08-01"},
        ).json()
        six_months = client.get(
            "/api/v1/dashboard",
            params={"start_date": "2024-02-01", "end_date": "2024-08-01"},
        ).json()
        selected_summary = client.get(
            "/api/v1/summary",
            params={"start_date": "2024-02-01", "end_date": "2024-08-01"},
        ).json()
        selected_port_history = client.get(
            "/api/v1/ports/uploaded-ais-area/history",
            params={"start_date": "2024-02-01", "end_date": "2024-08-01"},
        ).json()["observations"]

    assert one_month["summary"]["observations"] == 32
    assert six_months["summary"]["observations"] == 183
    assert len(one_month["history"]) == 32
    assert len(six_months["history"]) == 183
    assert six_months["summary"]["observations"] == selected_summary["observations"]
    assert six_months["summary"]["coverage"] == {
        "start": "2024-02-01",
        "end": "2024-08-01",
    }
    assert six_months["summary"]["congestion_days"] is None
    assert [row["date"] for row in six_months["history"]][0] == "2024-02-01"
    assert six_months["history"] == selected_port_history


def test_long_date_filter_does_not_extend_one_month_dataset(tmp_path, monkeypatch):
    configure_dataset_paths(tmp_path, monkeypatch)
    dates = pd.date_range("2025-05-01", "2025-05-31", freq="D").strftime("%Y-%m-%d").tolist()

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("one-month.csv", ais_csv_for_dates(dates), "text/csv")},
        )
        long_range = client.get(
            "/api/v1/dashboard",
            params={"start_date": "2024-12-01", "end_date": "2025-05-31"},
        ).json()

    assert upload.status_code == 200, upload.text
    assert len(long_range["history"]) == 31
    assert long_range["summary"]["observations"] == 31
    assert long_range["summary"]["coverage"] == {
        "start": "2025-05-01",
        "end": "2025-05-31",
    }


def test_upload_switches_all_analytics_to_new_dataset_and_survives_reload(
    tmp_path, monkeypatch
):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    uploaded_csv = ais_csv(
        anchored_vessels=31,
        days=("2025-05-01", "2025-05-02"),
        vessel_count=40,
    )
    original_metrics = paths["metrics"].read_bytes()

    with TestClient(app) as client:
        saved_metrics_before_upload = client.get("/api/v1/metrics").json()
        original_summary = client.get("/api/v1/summary").json()
        assert original_summary["coverage"] == {
            "start": "2024-01-02",
            "end": "2024-01-02",
        }

        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("may-observations.csv", uploaded_csv, "text/csv")},
            data={"source_name": "Identifiable May fixture"},
        )
        assert upload.status_code == 200, upload.text
        uploaded = upload.json()

        summary = client.get("/api/v1/summary").json()
        date_filtered_summary = client.get(
            "/api/v1/summary",
            params={"start_date": "2025-05-02", "end_date": "2025-05-02"},
        ).json()
        history = client.get(
            "/api/v1/ports/port-of-los-angeles/history"
        ).json()["observations"]
        date_filtered_history = client.get(
            "/api/v1/ports/port-of-los-angeles/history",
            params={"start_date": "2025-05-02", "end_date": "2025-05-02"},
        ).json()["observations"]
        snapshot = client.get("/api/v1/dashboard").json()
        active = client.get("/api/v1/datasets/active").json()
        saved_metrics_after_upload = client.get("/api/v1/metrics").json()

        invalid_upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("invalid.csv", b"LAT,LON\n33.7,-118.4\n", "text/csv")},
        )
        summary_after_invalid = client.get("/api/v1/summary").json()
        snapshot_after_invalid = client.get("/api/v1/dashboard").json()

    expected = {
        "date": "2025-05-02",
        "vessel_density": 40,
        "anchorage_vessels": 31,
        "next_day_congestion": 1,
    }
    assert uploaded["date_coverage"] == {"start": "2025-05-02", "end": "2025-05-02"}
    assert uploaded["sha256"] == active["sha256"]
    assert active["source_name"] == "Identifiable May fixture"
    assert active["processing_status"] == "processed"
    assert active["processed_rows"] == 1
    assert summary["coverage"] == uploaded["date_coverage"]
    assert summary["observations"] == 1
    assert summary["congestion_days"] == 1
    assert date_filtered_summary["observations"] == 1
    assert {key: history[0][key] for key in expected} == expected
    assert date_filtered_history == history
    assert {key: snapshot["history"][0][key] for key in expected} == expected
    assert snapshot["active_dataset"]["sha256"] == uploaded["sha256"]
    assert snapshot["active_dataset"]["processing_status"] == "processed"
    assert snapshot["metadata"]["source_name"] == "Identifiable May fixture"
    assert snapshot["metadata"]["model_training_dataset"]["data_status"] == "synthetic"
    assert snapshot["metadata"]["evaluation_dataset"]["data_status"] == "synthetic"
    assert snapshot["metrics"]["training_dataset"]["data_status"] == "synthetic"
    assert snapshot["metadata"]["evaluation_dataset_matches_active"] is False
    assert saved_metrics_after_upload == saved_metrics_before_upload
    assert snapshot["metrics"]["splits"]["test"]["period"] == {
        "start": "2024-10-18",
        "end": "2024-12-30",
    }
    assert invalid_upload.status_code == 422
    assert summary_after_invalid["coverage"] == uploaded["date_coverage"]
    assert snapshot_after_invalid["history"] == snapshot["history"]
    assert paths["metrics"].read_bytes() == original_metrics

    with TestClient(app) as client:
        reloaded = client.get("/api/v1/dashboard").json()
    assert reloaded["active_dataset"]["sha256"] == uploaded["sha256"]
    assert reloaded["history"] == snapshot["history"]


def test_upload_validates_and_processes_csv_transactionally(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    upload_dir = paths["processed"].parents[1] / "uploads"

    with TestClient(app) as client:
        validation = client.post(
            "/api/v1/datasets/validate",
            files={"file": ("observations.csv", ais_csv(), "text/csv")},
        )
        assert validation.status_code == 200
        assert validation.json()["valid"] is True
        assert validation.json()["input_rows"] == 120
        assert len(validation.json()["preview"]) == 10
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("observations.csv", ais_csv(), "text/csv")},
            data={"source_name": "AIS test upload"},
        )
        dashboard = client.get("/api/v1/dashboard")
        filtered_dashboard = client.get(
            "/api/v1/dashboard",
            params={"start_date": "2024-01-02", "end_date": "2024-01-02"},
        )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "processed"
    assert result["input_rows"] == 120
    assert result["processed_rows"] == 1
    assert result["date_coverage"] == {"start": "2024-01-02", "end": "2024-01-02"}
    assert result["model_retrained"] is False
    assert pd.read_csv(paths["processed"]).iloc[0]["next_day_congestion"] == 1
    assert json_load(paths["provenance"])["data_status"] == "uploaded"
    active = datasets.get_active_dataset()
    assert active["source_name"] == "AIS test upload"
    assert active["processed_rows"] == 1
    assert len(list(upload_dir.glob("*.csv"))) == 1
    assert dashboard.status_code == 200
    snapshot = dashboard.json()
    assert snapshot["metadata"]["source_name"] == "AIS test upload"
    assert snapshot["metadata"]["observation_start"] == "2024-01-02"
    assert snapshot["summary"]["observations"] == 1
    assert snapshot["summary"]["congestion_days"] == 1
    assert snapshot["history"][0]["date"] == "2024-01-02"
    assert snapshot["history"][0]["next_day_congestion"] == 1
    assert snapshot["metrics"]["test"]["sample_count"] == 74
    assert snapshot["metrics"]["training_dataset"]["data_status"] == "synthetic"
    assert snapshot["metadata"]["evaluation_dataset_matches_active"] is False
    filtered_snapshot = filtered_dashboard.json()
    assert filtered_snapshot["history"] == snapshot["history"]
    assert filtered_snapshot["summary"]["coverage"] == {
        "start": "2024-01-02",
        "end": "2024-01-02",
    }
    evaluation_before_reload = paths["metrics"].read_bytes()

    with TestClient(app) as client:
        reloaded = client.get("/api/v1/dashboard").json()
    assert reloaded["metadata"]["dataset_sha256"] == active["sha256"]
    assert reloaded["history"] == snapshot["history"]
    assert paths["metrics"].read_bytes() == evaluation_before_reload


def test_dashboard_reports_zero_events_only_when_active_rows_have_zero_labels(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    contents = ais_csv_for_dates(
        ["2024-01-01", "2024-01-02"],
        latitude=33.74,
        longitude=-118.4,
        vessel_count=30,
        synthetic_label=1,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("no-events.csv", contents, "text/csv")},
        )
        snapshot = client.get("/api/v1/dashboard").json()
        summary = client.get("/api/v1/summary").json()
        port_history = client.get(
            "/api/v1/ports/port-of-los-angeles/history"
        ).json()["observations"]

    assert response.status_code == 200
    assert snapshot["summary"]["observations"] == 1
    assert snapshot["summary"]["congestion_days"] == 0
    assert snapshot["summary"]["congestion_rate"] == 0
    assert snapshot["summary"]["target_available"] is True
    assert snapshot["metadata"]["target_status"] == "derived_proxy"
    assert snapshot["metadata"]["target_source"].startswith("Derived from AIS messages")
    assert snapshot["history"][0]["next_day_congestion"] == 0
    assert snapshot["summary"]["congestion_days"] == summary["congestion_days"] == 0
    assert snapshot["summary"]["congestion_rate"] == summary["congestion_rate"] == 0
    assert snapshot["history"] == port_history
    assert json_load(paths["quality"])["provided_target_columns_ignored"] == [
        "SyntheticCongestionLabel"
    ]


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_upload_rejects_missing_required_columns_and_preserves_active_data(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    old_content = paths["processed"].read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("bad.csv", b"LAT,LON\n33.7,-118.4\n", "text/csv")},
        )
    assert response.status_code == 422
    assert "MMSI" in response.json()["detail"]["validation"]["missing_columns"]
    assert paths["processed"].read_bytes() == old_content
    assert json_load(paths["provenance"])["data_status"] == "synthetic"


def test_validation_rejects_empty_and_non_csv_uploads():
    with TestClient(app) as client:
        empty = client.post(
            "/api/v1/datasets/validate",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        wrong_type = client.post(
            "/api/v1/datasets/validate",
            files={"file": ("observations.txt", ais_csv(), "text/plain")},
        )
    assert empty.status_code == 422
    assert "empty" in empty.json()["detail"]["validation"]["errors"][0].lower()
    assert wrong_type.status_code == 422
    assert ".csv" in wrong_type.json()["detail"]["validation"]["errors"][0]


def test_validation_reports_missing_values_invalid_dates_numeric_data_and_duplicates():
    frame = pd.read_csv(BytesIO(ais_csv()), dtype=str)
    frame.loc[0, "MMSI"] = None
    frame.loc[1, "BaseDateTime"] = "not-a-date"
    frame.loc[2, "LAT"] = "north"
    frame.loc[3, "LON"] = "200"
    frame.loc[4, "SOG"] = "-1"
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    frame.loc[6, "MMSI"] = "invalid-id"
    report = datasets.validate_frame(frame)
    assert report["valid"] is False
    assert report["invalid_mmsi_values"] == 1
    assert report["invalid_dates"] == 1
    assert report["invalid_numeric_values"] == {"LAT": 1, "LON": 1, "SOG": 1}
    assert report["duplicate_records"] == 1
    assert report["warnings"]


def test_validation_rejects_malformed_csv():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/validate",
            files={
                "file": (
                    "malformed.csv",
                    b'MMSI,BaseDateTime,LAT,LON,SOG\n"unterminated,2024-01-01,33.7,-118.4,1',
                    "text/csv",
                )
            },
        )
    assert response.status_code == 422
    assert "parse" in response.json()["detail"]["validation"]["errors"][0].lower()


def test_upload_enforces_byte_limit_and_closes_without_replacement(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(datasets, "MAX_UPLOAD_BYTES", 8)
    original = paths["processed"].read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("large.csv", ais_csv(), "text/csv")},
        )
    assert response.status_code == 413
    assert paths["processed"].read_bytes() == original
    assert json_load(paths["provenance"])["data_status"] == "synthetic"


def test_processing_failure_leaves_existing_dataset_untouched(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    original = {key: path.read_bytes() for key, path in paths.items()}

    def fail_processing(frame):
        raise OSError("simulated processing failure")

    monkeypatch.setattr(datasets, "build_daily_dataset", fail_processing)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("new.csv", ais_csv(), "text/csv")},
        )
    assert response.status_code == 500
    assert "active dataset was not replaced" in response.json()["detail"]
    assert {key: path.read_bytes() for key, path in paths.items()} == original
    assert not list((paths["processed"].parents[1] / "uploads").glob("*.csv"))


def test_failed_file_commit_rolls_back_previous_dataset(tmp_path, monkeypatch):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    original = {key: path.read_bytes() for key, path in paths.items()}
    replace = datasets.os.replace

    def fail_quality_replace(source, destination):
        if Path(destination) == paths["quality"]:
            raise OSError("simulated atomic replace failure")
        replace(source, destination)

    monkeypatch.setattr(datasets.os, "replace", fail_quality_replace)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("new.csv", ais_csv(), "text/csv")},
        )
    assert response.status_code == 500
    assert {key: path.read_bytes() for key, path in paths.items()} == original
    assert not list((paths["processed"].parents[1] / "uploads").glob("*.csv"))


def test_date_filtered_summary_and_history_use_inclusive_actual_dates(tmp_path, monkeypatch):
    data_file = tmp_path / "daily.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]),
            "vessel_density": [10, 20],
            "anchorage_vessels": [2, 5],
            "anchorage_dwell_hours": [1.0, 2.0],
            "day_of_week": [0, 1],
            "month": [1, 1],
            "next_day_congestion": [0, 1],
        }
    )
    frame.to_csv(data_file, index=False)
    monkeypatch.setattr(runtime, "PROCESSED_PATH", data_file)
    monkeypatch.setattr(runtime, "QUALITY_PATH", tmp_path / "missing-quality.json")
    monkeypatch.setattr(runtime, "PROVENANCE_PATH", tmp_path / "missing-provenance.json")
    with TestClient(app) as client:
        summary = client.get(
            "/api/v1/summary",
            params={"start_date": "2024-01-02", "end_date": "2024-01-02"},
        )
        history = client.get(
            "/api/v1/ports/port-of-los-angeles/history",
            params={"start_date": "2024-01-02", "end_date": "2024-01-02"},
        )
        empty = client.get(
            "/api/v1/summary",
            params={"start_date": "2025-01-01", "end_date": "2025-02-01"},
        )
        invalid = client.get(
            "/api/v1/summary",
            params={"start_date": "2024-02-01", "end_date": "2024-01-01"},
        )
    assert summary.status_code == history.status_code == empty.status_code == 200
    assert summary.json()["observations"] == 1
    assert summary.json()["congestion_days"] == 1
    assert history.json()["observations"][0]["date"] == "2024-01-02"
    assert empty.json()["dataset_available"] is True
    assert empty.json()["data_available"] is False
    assert invalid.status_code == 422


def test_summary_metrics_are_recalculated_for_each_date_range(tmp_path, monkeypatch):
    data_file = tmp_path / "daily.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                ],
                utc=True,
            ),
            "vessel_density": [10, 20, 30, 40],
            "anchorage_vessels": [2, 0, 3, 4],
            "anchorage_dwell_hours": [1.0, None, 3.0, 4.0],
            "day_of_week": [0, 1, 2, 3],
            "month": [1, 1, 1, 1],
            "next_day_congestion": [0, 1, None, 1],
        }
    )
    frame.to_csv(data_file, index=False)
    monkeypatch.setattr(runtime, "PROCESSED_PATH", data_file)
    monkeypatch.setattr(runtime, "QUALITY_PATH", tmp_path / "missing-quality.json")
    monkeypatch.setattr(runtime, "PROVENANCE_PATH", tmp_path / "missing-provenance.json")

    with TestClient(app) as client:
        first_params = {"start_date": "2024-01-01", "end_date": "2024-01-02"}
        second_params = {"start_date": "2024-01-02", "end_date": "2024-01-03"}
        last_params = {"start_date": "2024-01-04", "end_date": "2024-01-04"}
        missing_label_params = {"start_date": "2024-01-03", "end_date": "2024-01-03"}
        no_data_params = {"start_date": "2025-01-01", "end_date": "2025-01-02"}
        first = client.get("/api/v1/summary", params=first_params).json()
        second = client.get("/api/v1/summary", params=second_params).json()
        last = client.get("/api/v1/summary", params=last_params).json()
        missing_labels = client.get(
            "/api/v1/summary", params=missing_label_params
        ).json()
        no_data = client.get("/api/v1/summary", params=no_data_params).json()
        repeated = client.get("/api/v1/summary", params=first_params).json()

    assert (first["observations"], first["target_observations"], first["congestion_days"], first["congestion_rate"]) == (2, 2, 1, 0.5)
    assert (second["observations"], second["target_observations"], second["congestion_days"], second["congestion_rate"]) == (2, 1, 1, 1.0)
    assert (last["observations"], last["target_observations"], last["congestion_days"], last["congestion_rate"]) == (1, 1, 1, 1.0)
    assert missing_labels["observations"] == 1
    assert missing_labels["target_observations"] == 0
    assert missing_labels["congestion_days"] is None
    assert missing_labels["congestion_rate"] is None
    assert no_data["data_available"] is False
    assert no_data["observations"] == 0
    assert no_data["congestion_days"] is None
    assert repeated == first


def configure_dataset_paths(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True)
    processed = data_dir / "processed" / "port_daily.csv"
    quality = data_dir / "processed" / "data_quality.json"
    provenance = data_dir / "manifests" / "provenance.json"
    processed.parent.mkdir(parents=True)
    provenance.parent.mkdir(parents=True)
    previous_daily, previous_quality = datasets.build_daily_dataset(
        pd.read_csv(BytesIO(ais_csv()))
    )
    previous_daily.to_csv(processed, index=False)
    quality.write_text(json.dumps(previous_quality), encoding="utf-8")
    provenance.write_text(
        json.dumps(
            {
                "data_status": "synthetic",
                "source_name": "Previous synthetic fixture",
                "filename": "previous.csv",
                "sha256": "previous-dataset-hash",
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "artifacts" / "metrics" / "evaluation.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        '{"data_status":"synthetic","training_dataset":{"data_status":"synthetic","sha256":"original-training-hash"},'
        '"test":{"sample_count":74},"splits":{"test":{"period":{"start":"2024-10-18","end":"2024-12-30"}}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets, "PROCESSED_PATH", processed)
    monkeypatch.setattr(datasets, "QUALITY_PATH", quality)
    monkeypatch.setattr(datasets, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(runtime, "PROCESSED_PATH", processed)
    monkeypatch.setattr(runtime, "QUALITY_PATH", quality)
    monkeypatch.setattr(runtime, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(runtime, "METRICS_PATH", metrics)
    return {
        "processed": processed,
        "quality": quality,
        "provenance": provenance,
        "metrics": metrics,
    }


def test_synthetic_dataset_upload_keeps_synthetic_provenance_and_dataset_identity(
    tmp_path, monkeypatch
):
    paths = configure_dataset_paths(tmp_path, monkeypatch)
    frame = pd.read_csv(
        BytesIO(
            ais_csv(
                anchored_vessels=8,
                days=("2025-01-01", "2025-01-02", "2025-01-03"),
            )
        )
    )
    frame["MMSI"] = frame["MMSI"].astype(int).astype(str).str.zfill(9)
    frame["SyntheticData"] = True
    frame["SyntheticCongestionLabel"] = frame["BaseDateTime"].str.startswith(
        "2025-01-02"
    ).astype(int)
    frame["SyntheticDelayLabel"] = frame["BaseDateTime"].str.startswith(
        "2025-01-02"
    ).astype(int)
    contents = frame.to_csv(index=False).encode("utf-8")

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("synthetic-development.csv", contents, "text/csv")},
            data={"source_name": "Generated fixture"},
        )
        active = client.get("/api/v1/datasets/active").json()

    assert upload.status_code == 200, upload.text
    assert active["data_status"] == "synthetic"
    assert active["synthetic"] is True
    provenance = json_load(paths["provenance"])
    quality = json_load(paths["quality"])
    assert provenance["sha256"] == upload.json()["sha256"]
    assert quality["source_dataset_sha256"] == upload.json()["sha256"]
    assert "synthetic" in active["source_name"].lower()
