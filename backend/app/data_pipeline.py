import pandas as pd

REQUIRED_COLUMNS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG"]
TARGET_DEFINITION = (
    "Next UTC day has at least 25 distinct MMSIs observed in the candidate-anchorage geofence."
)
ANCHORAGE_THRESHOLD = 25
PORT_BOUNDS = {"lat_min": 33.6, "lat_max": 34.2, "lon_min": -118.6, "lon_max": -118.0}


def inspect_ais(frame: pd.DataFrame) -> dict:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    report = {
        "input_rows": int(len(frame)),
        "original_columns": list(frame.columns),
        "missing_columns": missing_columns,
        "missingness": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        },
        "duplicate_messages": 0,
        "invalid_timestamps": 0,
        "invalid_coordinates": 0,
        "errors": [],
    }
    if missing_columns:
        report["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
        return report
    timestamps = pd.to_datetime(
        frame["BaseDateTime"], errors="coerce", utc=True, format="mixed"
    )
    report["invalid_timestamps"] = int(timestamps.isna().sum())
    lat = pd.to_numeric(frame["LAT"], errors="coerce")
    lon = pd.to_numeric(frame["LON"], errors="coerce")
    invalid_coords = (
        lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)
    )
    report["invalid_coordinates"] = int(invalid_coords.sum())
    normalized = pd.DataFrame(
        {"MMSI": frame["MMSI"], "timestamp": timestamps}
    )
    report["duplicate_messages"] = int(
        normalized.duplicated(["MMSI", "timestamp"]).sum()
    )
    return report


def build_daily_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    quality = inspect_ais(frame)
    if quality["errors"]:
        raise ValueError("; ".join(quality["errors"]))
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(
        data["BaseDateTime"], errors="coerce", utc=True, format="mixed"
    )
    for column in ("LAT", "LON", "SOG"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    original_count = len(data)
    data = data.dropna(subset=["MMSI", "timestamp", "LAT", "LON", "SOG"])
    valid_coords = data["LAT"].between(-90, 90) & data["LON"].between(-180, 180)
    data = data.loc[valid_coords]
    before_dedupe = len(data)
    data = data.drop_duplicates(["MMSI", "timestamp"], keep="last")
    deduplicated = before_dedupe - len(data)
    in_region = (
        data["LAT"].between(PORT_BOUNDS["lat_min"], PORT_BOUNDS["lat_max"])
        & data["LON"].between(PORT_BOUNDS["lon_min"], PORT_BOUNDS["lon_max"])
    )
    data = data.loc[in_region].copy()
    data["date"] = data["timestamp"].dt.floor("D")
    data["candidate_anchorage"] = (
        data["LAT"].between(33.68, 33.82)
        & data["LON"].between(-118.49, -118.31)
        & data["SOG"].between(0, 0.5)
    )
    data["anchorage_mmsi"] = data["MMSI"].where(data["candidate_anchorage"])
    vessel_days = data.groupby(["date", "MMSI"], as_index=False).agg(
        anchored=("candidate_anchorage", "max"),
        anchorage_observations=("candidate_anchorage", "sum"),
        first_anchorage=(
            "timestamp",
            lambda values: values[data.loc[values.index, "candidate_anchorage"]].min()
            if data.loc[values.index, "candidate_anchorage"].any()
            else pd.NaT,
        ),
        last_anchorage=(
            "timestamp",
            lambda values: values[data.loc[values.index, "candidate_anchorage"]].max()
            if data.loc[values.index, "candidate_anchorage"].any()
            else pd.NaT,
        ),
    )
    vessel_days["dwell_hours"] = (
        vessel_days["last_anchorage"] - vessel_days["first_anchorage"]
    ).dt.total_seconds() / 3600
    vessel_days.loc[vessel_days["anchorage_observations"] < 2, "dwell_hours"] = float("nan")
    daily_counts = data.groupby("date").agg(
        vessel_density=("MMSI", "nunique"),
        anchorage_vessels=("anchorage_mmsi", "nunique"),
    )
    dwell = (
        vessel_days.loc[
            vessel_days["anchored"] & vessel_days["dwell_hours"].notna()
        ]
        .groupby("date")["dwell_hours"]
        .median()
        .rename("anchorage_dwell_hours")
    )
    daily = daily_counts.join(dwell, how="left").sort_index()
    next_date = daily.index + pd.Timedelta(days=1)
    next_counts = daily["anchorage_vessels"].reindex(next_date).to_numpy()
    has_next_day = (next_date <= daily.index.max()) & ~pd.isna(next_counts)
    daily = daily.loc[has_next_day].copy()
    next_counts = next_counts[has_next_day]
    daily["date"] = daily.index + pd.Timedelta(days=1)
    daily["day_of_week"] = daily["date"].dt.dayofweek.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)
    daily["ais_congestion_proxy"] = (next_counts >= ANCHORAGE_THRESHOLD).astype(int)
    daily["synthetic_congestion_label"] = float("nan")
    synthetic_label_available = "SyntheticCongestionLabel" in data.columns
    if synthetic_label_available:
        provided_labels = data["SyntheticCongestionLabel"]
        synthetic_values = pd.to_numeric(provided_labels, errors="coerce")
        invalid_synthetic = provided_labels.notna() & (
            synthetic_values.isna() | ~synthetic_values.isin([0, 1])
        )
        if invalid_synthetic.any():
            raise ValueError(
                "SyntheticCongestionLabel contains non-binary values; expected 0/1 labels."
            )
        data["SyntheticCongestionLabel"] = synthetic_values
        aggregated_labels = (
            data.groupby("date")["SyntheticCongestionLabel"]
            .max(min_count=1)
            .rename("next_day_congestion")
            .reindex(daily["date"])
        )
        daily["synthetic_congestion_label"] = aggregated_labels.to_numpy()
        if aggregated_labels.nunique() >= 2:
            daily["next_day_congestion"] = aggregated_labels.to_numpy()
            quality["target_status"] = "provided"
            quality["target_source"] = (
                "SyntheticCongestionLabel column in the source dataset; aggregated by UTC day using max value."
            )
            quality["target_definition"] = (
                "The uploaded synthetic congestion label is used as the binary target when it is supplied "
                "and valid."
            )
            quality["provided_target_columns_used"] = ["SyntheticCongestionLabel"]
            quality["provided_target_columns_ignored"] = []
        else:
            daily["next_day_congestion"] = daily["ais_congestion_proxy"].astype(int)
            quality["target_status"] = "derived_proxy"
            quality["target_source"] = (
                "Derived from AIS messages using the configured Port of Los Angeles candidate-anchorage geofence."
            )
            quality["target_definition"] = TARGET_DEFINITION
            quality["provided_target_columns_used"] = []
            quality["provided_target_columns_ignored"] = ["SyntheticCongestionLabel"]
    else:
        daily["next_day_congestion"] = daily["ais_congestion_proxy"].astype(int)
        quality["target_status"] = "derived_proxy"
        quality["target_source"] = (
            "Derived from AIS messages using the configured Port of Los Angeles candidate-anchorage geofence."
        )
        quality["target_definition"] = TARGET_DEFINITION
        quality["provided_target_columns_used"] = []
        quality["provided_target_columns_ignored"] = []
    daily = daily.reset_index(drop=True)[
        [
            "date",
            "vessel_density",
            "anchorage_vessels",
            "anchorage_dwell_hours",
            "ais_congestion_proxy",
            "synthetic_congestion_label",
            "day_of_week",
            "month",
            "next_day_congestion",
        ]
    ]
    measured_dwell = daily["anchorage_dwell_hours"].dropna()
    quality["candidate_anchorage_days"] = int(
        (daily["anchorage_vessels"] > 0).sum()
    )
    quality["dwell_measurement_days"] = int(len(measured_dwell))
    quality["anchorage_days_without_sufficient_observations"] = int(
        ((daily["anchorage_vessels"] > 0) & daily["anchorage_dwell_hours"].isna()).sum()
    )
    quality["dwell_coverage"] = (
        {
            "start": daily.loc[
                daily["anchorage_dwell_hours"].notna(), "date"
            ].min().date().isoformat(),
            "end": daily.loc[
                daily["anchorage_dwell_hours"].notna(), "date"
            ].max().date().isoformat(),
        }
        if not measured_dwell.empty
        else None
    )
    quality["congestion_label_distribution"] = {
        "not_congested": int((daily["next_day_congestion"] == 0).sum()),
        "congested": int((daily["next_day_congestion"] == 1).sum()),
    }
    quality.update(
        {
            "valid_rows_before_region_filter": int(original_count - quality["invalid_timestamps"] - quality["invalid_coordinates"]),
            "duplicate_rows_removed": int(deduplicated),
            "rows_after_region_filter": int(len(data)),
            "rows_removed_as_missing_or_invalid": int(original_count - before_dedupe),
            "region": PORT_BOUNDS,
            "candidate_anchorage_rule": "33.68-33.82 N, -118.49 to -118.31 W, SOG 0-0.5 knots",
            "label_threshold_vessels": ANCHORAGE_THRESHOLD,
            "label_definition": TARGET_DEFINITION,
            "missing_dwell_policy": "Dwell is unavailable when no qualifying vessel has at least two qualifying observations that day; it is not filled with zero.",
            "processed_rows": int(len(daily)),
            "positive_labels": int(daily["next_day_congestion"].sum()),
            "negative_labels": int((daily["next_day_congestion"] == 0).sum()),
            "synthetic_congestion_label_distribution": {
                "not_congested": int((daily["synthetic_congestion_label"] == 0).sum()),
                "congested": int((daily["synthetic_congestion_label"] == 1).sum()),
                "unlabeled": int(daily["synthetic_congestion_label"].isna().sum()),
            },
            "observation_start": daily["date"].min().date().isoformat() if not daily.empty else None,
            "observation_end": daily["date"].max().date().isoformat() if not daily.empty else None,
        }
    )
    return daily, quality


def build_delay_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base_daily, quality = build_daily_dataset(frame)
    if base_daily.empty:
        return base_daily, quality
    quality["shipment_delay_definition"] = (
        "SyntheticDelayLabel aggregated by UTC target day using max; the model uses "
        "the preceding UTC day's operational inputs. No labels are derived from delay minutes."
    )
    if "SyntheticDelayLabel" not in frame.columns:
        quality["delay_target_status"] = "unavailable"
        quality["delay_target_source"] = None
        return base_daily, quality

    raw = frame.copy()
    raw["timestamp"] = pd.to_datetime(
        raw["BaseDateTime"], errors="coerce", utc=True, format="mixed"
    )
    raw["date"] = raw["timestamp"].dt.floor("D")
    for column in ("LAT", "LON", "SOG"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["MMSI", "timestamp", "LAT", "LON", "SOG"])
    raw = raw.loc[
        raw["LAT"].between(PORT_BOUNDS["lat_min"], PORT_BOUNDS["lat_max"])
        & raw["LON"].between(PORT_BOUNDS["lon_min"], PORT_BOUNDS["lon_max"])
    ].drop_duplicates(["MMSI", "timestamp"], keep="last")

    provided_labels = raw["SyntheticDelayLabel"]
    label_values = pd.to_numeric(provided_labels, errors="coerce")
    invalid_labels = provided_labels.notna() & (
        label_values.isna() | ~label_values.isin([0, 1])
    )
    if invalid_labels.any():
        raise ValueError("SyntheticDelayLabel contains non-binary values; expected 0/1 labels.")
    raw["SyntheticDelayLabel"] = label_values

    def daily_mode(values: pd.Series) -> str:
        modes = values.dropna().mode()
        return str(modes.iloc[0]) if not modes.empty else "Unknown"

    raw["vessel_arrivals_daily"] = (
        pd.to_numeric(raw["VesselArrivals_Daily"], errors="coerce")
        if "VesselArrivals_Daily" in raw.columns
        else raw.groupby("date")["MMSI"].transform("nunique")
    )
    raw["queue_length_vessels"] = (
        pd.to_numeric(raw["QueueLength_Vessels"], errors="coerce")
        if "QueueLength_Vessels" in raw.columns
        else float("nan")
    )
    raw["port_capacity_berths"] = (
        pd.to_numeric(raw["PortCapacity_Berths"], errors="coerce")
        if "PortCapacity_Berths" in raw.columns
        else float("nan")
    )
    if "WindSpeed_kmh" in raw.columns:
        raw["wind_speed_kmh"] = pd.to_numeric(raw["WindSpeed_kmh"], errors="coerce")
    elif "WindSpeed_knots" in raw.columns:
        raw["wind_speed_kmh"] = pd.to_numeric(raw["WindSpeed_knots"], errors="coerce") * 1.852
    else:
        raw["wind_speed_kmh"] = float("nan")
    raw["weather_condition"] = (
        raw["WeatherCondition"].fillna("Unknown").astype(str)
        if "WeatherCondition" in raw.columns
        else "Unknown"
    )

    operational = raw.groupby("date").agg(
        vessel_arrivals_daily=("vessel_arrivals_daily", "median"),
        queue_length_vessels=("queue_length_vessels", "median"),
        port_capacity_berths=("port_capacity_berths", "median"),
        weather_condition=("weather_condition", daily_mode),
        wind_speed_kmh=("wind_speed_kmh", "mean"),
    )
    # Inputs are measured on the day before the target label is observed.
    operational.index = operational.index + pd.Timedelta(days=1)
    delay_data = base_daily.merge(
        operational, left_on="date", right_index=True, how="left"
    )
    labels = (
        raw.groupby("date")["SyntheticDelayLabel"]
        .max(min_count=1)
        .rename("shipment_delay")
    )
    delay_data = delay_data.merge(labels, left_on="date", right_index=True, how="left")
    delay_data["weather_condition"] = delay_data["weather_condition"].fillna("Unknown")
    valid_labels = delay_data["shipment_delay"].dropna().astype(int)
    quality["delay_target_status"] = (
        "provided" if valid_labels.nunique() == 2 else "unavailable"
    )
    quality["delay_target_source"] = "SyntheticDelayLabel in the source dataset, aggregated by UTC day using max."
    quality["delay_target_distribution"] = {
        "not_delayed": int((valid_labels == 0).sum()),
        "delayed": int((valid_labels == 1).sum()),
    }
    return delay_data[
        [
            "date",
            "vessel_density",
            "anchorage_vessels",
            "anchorage_dwell_hours",
            "vessel_arrivals_daily",
            "queue_length_vessels",
            "port_capacity_berths",
            "weather_condition",
            "wind_speed_kmh",
            "day_of_week",
            "month",
            "ais_congestion_proxy",
            "synthetic_congestion_label",
            "next_day_congestion",
            "shipment_delay",
        ]
    ], quality


def build_unlabeled_daily_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    quality = inspect_ais(frame)
    if quality["errors"]:
        raise ValueError("; ".join(quality["errors"]))

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(
        data["BaseDateTime"], errors="coerce", utc=True, format="mixed"
    )
    for column in ("LAT", "LON", "SOG"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    original_count = len(data)
    data = data.dropna(subset=["MMSI", "timestamp", "LAT", "LON", "SOG"])
    valid_coords = data["LAT"].between(-90, 90) & data["LON"].between(-180, 180)
    data = data.loc[valid_coords]
    before_dedupe = len(data)
    data = data.drop_duplicates(["MMSI", "timestamp"], keep="last")
    deduplicated = before_dedupe - len(data)
    data["date"] = data["timestamp"].dt.floor("D")

    daily = data.groupby("date").agg(vessel_density=("MMSI", "nunique")).sort_index()
    daily["anchorage_vessels"] = float("nan")
    daily["anchorage_dwell_hours"] = float("nan")
    daily["date"] = daily.index
    daily["day_of_week"] = daily.index.dayofweek.astype(int)
    daily["month"] = daily.index.month.astype(int)
    daily["next_day_congestion"] = float("nan")
    daily = daily.reset_index(drop=True)[
        [
            "date",
            "vessel_density",
            "anchorage_vessels",
            "anchorage_dwell_hours",
            "day_of_week",
            "month",
            "next_day_congestion",
        ]
    ]
    quality.update(
        {
            "valid_rows_before_region_filter": int(
                original_count
                - quality["invalid_timestamps"]
                - quality["invalid_coordinates"]
            ),
            "duplicate_rows_removed": int(deduplicated),
            "rows_after_geographic_filter": int(len(data)),
            "rows_removed_as_missing_or_invalid": int(original_count - before_dedupe),
            "target_status": "unavailable",
            "target_source": None,
            "target_definition": (
                "No congestion proxy is derived because the uploaded coordinates do not "
                "match the configured Port of Los Angeles development geofence."
            ),
            "label_definition": None,
            "processed_rows": int(len(daily)),
            "positive_labels": None,
            "negative_labels": None,
            "observation_start": (
                daily["date"].min().date().isoformat() if not daily.empty else None
            ),
            "observation_end": (
                daily["date"].max().date().isoformat() if not daily.empty else None
            ),
        }
    )
    return daily, quality
