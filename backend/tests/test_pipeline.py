import pandas as pd
import pytest

from app.data_pipeline import build_daily_dataset, build_delay_dataset, inspect_ais


def example_messages():
    rows = []
    for day, anchored_count in (("2024-01-01", 2), ("2024-01-02", 3)):
        for vessel in range(30):
            anchored = vessel < anchored_count
            for hour in (6, 18):
                rows.append(
                    {
                        "MMSI": vessel + 100,
                        "BaseDateTime": f"{day}T{hour:02}:00:00Z",
                        "LAT": 33.74 if anchored else 33.9,
                        "LON": -118.4 if anchored else -118.2,
                        "SOG": 0.2 if anchored else 5,
                    }
                )
    return pd.DataFrame(rows)


def test_required_columns_and_bad_values_are_reported():
    assert inspect_ais(pd.DataFrame({"LAT": [1]}))["errors"]
    frame = example_messages()
    frame.loc[0, "BaseDateTime"] = "not a time"
    frame.loc[1, "LAT"] = 100
    result = inspect_ais(frame)
    assert result["invalid_timestamps"] == 1
    assert result["invalid_coordinates"] == 1


def test_daily_target_uses_next_calendar_day_and_skips_gap():
    frame = example_messages()
    daily, quality = build_daily_dataset(frame)
    assert daily.iloc[0]["date"] == pd.Timestamp("2024-01-02", tz="UTC")
    assert daily.iloc[0]["next_day_congestion"] == 0
    assert quality["duplicate_rows_removed"] == 0
    with_gap = frame.loc[~frame["BaseDateTime"].str.startswith("2024-01-02")]
    empty, _ = build_daily_dataset(with_gap)
    assert empty.empty


def test_duplicate_messages_removed_deterministically():
    frame = example_messages()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    first, quality = build_daily_dataset(duplicated)
    second, _ = build_daily_dataset(duplicated)
    pd.testing.assert_frame_equal(first, second)
    assert quality["duplicate_rows_removed"] == 1


def test_missing_and_out_of_bounds_rows_are_excluded():
    frame = example_messages()
    frame.loc[0, "MMSI"] = None
    frame.loc[1, "LAT"] = 95
    frame.loc[2, "LAT"] = 40
    daily, quality = build_daily_dataset(frame)
    assert daily["vessel_density"].max() <= 30
    assert quality["rows_removed_as_missing_or_invalid"] >= 2


def test_temporal_features_are_from_prediction_day():
    daily, quality = build_daily_dataset(example_messages())
    assert daily.iloc[0]["day_of_week"] == 1
    assert daily.iloc[0]["month"] == 1
    assert daily.iloc[0]["anchorage_vessels"] == 2
    assert daily.iloc[0]["anchorage_dwell_hours"] == 12
    assert quality["dwell_measurement_days"] == 1
    assert "MMSI" not in daily.columns


def test_single_qualifying_observation_has_unavailable_dwell():
    rows = []
    for day in ("2024-01-01", "2024-01-02"):
        rows.extend(
            [
                {
                    "MMSI": "367000001",
                    "BaseDateTime": f"{day}T06:00:00Z",
                    "LAT": 33.74,
                    "LON": -118.4,
                    "SOG": 0.2 if day == "2024-01-01" else 5,
                },
                {
                    "MMSI": "367000001",
                    "BaseDateTime": f"{day}T18:00:00Z",
                    "LAT": 33.74,
                    "LON": -118.4,
                    "SOG": 5,
                },
                {
                    "MMSI": "367000002",
                    "BaseDateTime": f"{day}T12:00:00Z",
                    "LAT": 33.9,
                    "LON": -118.2,
                    "SOG": 5,
                },
            ]
        )
    daily, quality = build_daily_dataset(pd.DataFrame(rows))

    assert daily.iloc[0]["anchorage_vessels"] == 1
    assert pd.isna(daily.iloc[0]["anchorage_dwell_hours"])
    assert quality["dwell_measurement_days"] == 0
    assert quality["anchorage_days_without_sufficient_observations"] == 1


def test_synthetic_congestion_label_is_separate_from_ais_proxy():
    frame = example_messages()
    frame["SyntheticCongestionLabel"] = frame["BaseDateTime"].str.startswith(
        "2024-01-02"
    ).astype(int)

    daily, quality = build_daily_dataset(frame)

    jan_second = daily.loc[daily["date"] == pd.Timestamp("2024-01-02", tz="UTC")].iloc[0]
    assert jan_second["synthetic_congestion_label"] == 1
    assert jan_second["next_day_congestion"] == 0
    assert jan_second["ais_congestion_proxy"] == 0
    assert quality["synthetic_congestion_label_distribution"]["congested"] == 1


def test_missing_required_column_fails_clearly():
    with pytest.raises(ValueError, match="Missing required columns"):
        build_daily_dataset(pd.DataFrame({"LAT": []}))


def test_delay_dataset_uses_supplied_label_and_previous_day_features_without_duration():
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for day_index in range(40):
        day = start + pd.Timedelta(days=day_index)
        label = int(day_index % 3 == 0)
        for vessel in range(3):
            rows.append(
                {
                    "MMSI": f"{day_index:06d}{vessel:03d}",
                    "BaseDateTime": (day + pd.Timedelta(hours=6)).isoformat(),
                    "LAT": 33.74,
                    "LON": -118.4,
                    "SOG": 0.2,
                    "VesselArrivals_Daily": day_index + 10,
                    "QueueLength_Vessels": day_index + 2,
                    "PortCapacity_Berths": 20,
                    "WeatherCondition": "Clear" if day_index % 2 else "Cloudy",
                    "WindSpeed_kmh": day_index,
                    "SyntheticDelayLabel": label,
                    "SyntheticDelayMinutes": 9000,
                }
            )
    daily, quality = build_delay_dataset(pd.DataFrame(rows))

    first = daily.iloc[0]
    assert first["date"] == start + pd.Timedelta(days=1)
    assert first["vessel_arrivals_daily"] == 10
    assert first["queue_length_vessels"] == 2
    assert first["shipment_delay"] == 0
    assert "SyntheticDelayMinutes" not in daily.columns
    assert "delay_duration_minutes" not in daily.columns
    assert quality["delay_target_status"] == "provided"
    assert quality["delay_target_distribution"]["not_delayed"] > 0
    assert quality["delay_target_distribution"]["delayed"] > 0


def test_delay_dataset_does_not_invent_labels_and_rejects_invalid_labels():
    frame = example_messages()
    no_labels, no_label_quality = build_delay_dataset(frame)
    assert "shipment_delay" not in no_labels.columns
    assert no_label_quality["delay_target_status"] == "unavailable"

    invalid = frame.assign(SyntheticDelayLabel=2)
    with pytest.raises(ValueError, match="SyntheticDelayLabel contains non-binary"):
        build_delay_dataset(invalid)
