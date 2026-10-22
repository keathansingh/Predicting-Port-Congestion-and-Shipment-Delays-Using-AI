import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic" / "ais_development.csv"
MANIFEST = ROOT / "data" / "manifests" / "provenance.json"
SEED = 42
LATENT_NOISE_PERSISTENCE = 0.8


def generate(
    output: Path = OUTPUT, days: int = 731, manifest_path: Path = MANIFEST
) -> Path:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-01", periods=days, freq="D", tz="UTC")
    rows = []
    latent_noise = 0.0
    previous_delay_minutes = 0.0
    fleet = np.arange(367_000_000, 367_000_160, dtype=np.int64)
    for index, date in enumerate(dates):
        seasonal = 8 * np.sin(2 * np.pi * index / 365.25)
        weekly = 5 * np.sin(2 * np.pi * index / 7)
        latent_noise = (
            LATENT_NOISE_PERSISTENCE * latent_noise
            + np.sqrt(1 - LATENT_NOISE_PERSISTENCE**2) * rng.normal(0, 9)
        )
        latent = 48 + seasonal + weekly + latent_noise
        vessel_count = int(np.clip(round(latent), 30, 70))
        month = date.month
        winter = month in (11, 12, 1, 2, 3)
        weather_options = ["Clear", "Cloudy", "Fog", "Rain", "Storm"]
        weather_weights = (
            [0.24, 0.27, 0.17, 0.24, 0.08]
            if winter
            else [0.40, 0.30, 0.15, 0.12, 0.03]
        )
        weather = str(rng.choice(weather_options, p=weather_weights))
        weather_pressure = {
            "Clear": 0.0,
            "Cloudy": 0.08,
            "Fog": 0.18,
            "Rain": 0.25,
            "Storm": 0.42,
        }[weather]
        wind_base = {"Clear": 9, "Cloudy": 14, "Fog": 10, "Rain": 24, "Storm": 42}[
            weather
        ]
        wind_speed = float(np.clip(rng.normal(wind_base, 7), 0, 85))
        annual_capacity = 24 + 2.5 * np.sin(2 * np.pi * (index - 80) / 365.25)
        port_capacity = int(np.clip(round(annual_capacity + rng.normal(0, 2)), 18, 32))
        vessel_arrivals = int(
            np.clip(vessel_count + rng.integers(-4, 5), 20, 80)
        )
        pressure_signal = (
            0.42 * vessel_arrivals / port_capacity
            + weather_pressure
            + max(0.0, wind_speed - 30) / 120
        )
        anchor_probability = float(
            np.clip(0.14 + 0.17 * pressure_signal + rng.normal(0, 0.035), 0.08, 0.48)
        )
        anchor_count = (
            0
            if rng.random() < 0.035
            else int(rng.binomial(vessel_count, anchor_probability))
        )
        queue_length = int(
            np.clip(
                round(
                    0.20 * vessel_arrivals
                    + 0.68 * anchor_count
                    + 0.42 * port_capacity * weather_pressure
                    + rng.normal(0, 3)
                ),
                0,
                100,
            )
        )
        queue_pressure = queue_length / port_capacity
        synthetic_congestion = int(
            queue_pressure >= 0.92
            or anchor_count >= 20
            or (weather in {"Rain", "Storm", "Fog"} and queue_pressure >= 0.72)
        )
        delay_minutes = float(
            np.clip(
                24
                + max(0.0, queue_pressure - 0.55) * 115
                + weather_pressure * 105
                + max(0.0, wind_speed - 20) * 0.45
                + rng.normal(0, 24),
                0,
                360,
            )
        )
        synthetic_delay = int(delay_minutes >= 60)
        vessel_ids = rng.choice(fleet, size=vessel_count, replace=False)
        anchored_ids = set(
            int(value)
            for value in rng.choice(vessel_ids, size=anchor_count, replace=False)
        )
        for vessel_index, raw_mmsi in enumerate(vessel_ids):
            mmsi = int(raw_mmsi)
            anchored = mmsi in anchored_ids
            if anchored:
                dwell_hours = float(
                    np.clip(
                        5
                        + 0.8 * queue_pressure
                        + 0.04 * (latent - 48)
                        + rng.normal(0, 3.5),
                        2,
                        20,
                    )
                )
                first_minute = int(rng.uniform(0, 24 * 60 - dwell_hours * 60))
                span_minutes = int(dwell_hours * 60)
                message_minutes = [first_minute, first_minute + span_minutes]
                if rng.random() < 0.2:
                    message_minutes.insert(
                        1, first_minute + int(span_minutes * rng.uniform(0.35, 0.65))
                    )
            else:
                message_minutes = [
                    int(rng.integers(5 * 60, 9 * 60)),
                    int(rng.integers(16 * 60, 21 * 60)),
                ]
            for message_index, minute in enumerate(message_minutes):
                timestamp = date + pd.Timedelta(minutes=int(minute))
                if anchored:
                    lat = rng.uniform(33.70, 33.79)
                    lon = rng.uniform(-118.46, -118.34)
                    sog = rng.uniform(0, 0.5)
                else:
                    lat = rng.uniform(33.84, 34.02)
                    lon = rng.uniform(-118.30, -118.08)
                    sog = rng.uniform(1.0, 12.0)
                rows.append(
                    {
                        "MMSI": mmsi,
                        "BaseDateTime": timestamp.isoformat(),
                        "LAT": round(lat, 6),
                        "LON": round(lon, 6),
                        "SOG": round(sog, 2),
                        "VesselArrivalTime": (
                            date + pd.Timedelta(minutes=message_minutes[0])
                        ).isoformat(),
                        "PortName": "Port of Los Angeles (synthetic development geofence)",
                        "PortCapacity_Berths": port_capacity,
                        "VesselArrivals_Daily": vessel_arrivals,
                        "QueueLength_Vessels": queue_length,
                        "WeatherCondition": weather,
                        "WindSpeed_kmh": round(wind_speed, 1),
                        "HistoricalDelayMinutes": round(previous_delay_minutes, 1),
                        "SyntheticCongestionLabel": synthetic_congestion,
                        "SyntheticDelayMinutes": round(delay_minutes, 1),
                        "SyntheticDelayLabel": synthetic_delay,
                        "SyntheticData": True,
                    }
                )
        previous_delay_minutes = delay_minutes
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "data_status": "synthetic",
                "source_name": "Seeded synthetic AIS-like development data",
                "source_url": None,
                "acquisition_date": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "filename": output.name,
                "file_size_bytes": output.stat().st_size,
                "sha256": checksum,
                "rows": len(frame),
                "observation_start": frame["BaseDateTime"].min(),
                "observation_end": frame["BaseDateTime"].max(),
                "columns": list(frame.columns),
                "original_columns": list(frame.columns),
                "synthetic": True,
                "synthetic_notice": "Generated observations; not real vessel broadcasts or public source data.",
                "geographic_scope": {
                    "port_id": "port-of-los-angeles",
                    "label": (
                        "Configured Port of Los Angeles synthetic development geofence "
                        "(not a real-world port identity claim)"
                    ),
                    "identified": True,
                    "identification_method": (
                        "Synthetic coordinates are generated inside the configured "
                        "development geofence; they do not verify a real port or AIS source."
                    ),
                    "coordinate_coverage": {
                        "latitude": {
                            "min": float(frame["LAT"].min()),
                            "max": float(frame["LAT"].max()),
                        },
                        "longitude": {
                            "min": float(frame["LON"].min()),
                            "max": float(frame["LON"].max()),
                        },
                    },
                    "geofence": {
                        "lat_min": 33.6,
                        "lat_max": 34.2,
                        "lon_min": -118.6,
                        "lon_max": -118.0,
                    },
                },
                "target_definitions": {
                    "SyntheticCongestionLabel": (
                        "Simulated congestion label driven by queue-to-capacity pressure, "
                        "anchorage count, and adverse weather."
                    ),
                    "SyntheticDelayLabel": (
                        "Simulated delay label derived from SyntheticDelayMinutes >= 60; "
                        "delay minutes depend on queue pressure, weather, wind, and seeded noise."
                    ),
                },
                "synthetic_process": {
                    "latent_noise_persistence": LATENT_NOISE_PERSISTENCE,
                    "anchorage_dwell_hours": (
                        "For qualifying low-speed geofence observations, repeated records for "
                        "the same vessel/day span a simulated duration; not ground-truth waiting time."
                    ),
                    "daily_conditions": (
                        "Arrivals, berth capacity, queue, anchorage share, weather, and wind "
                        "vary by season, traffic, and seeded random variation."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(frame):,} synthetic AIS-like records: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--days", type=int, default=731)
    args = parser.parse_args()
    generate(args.output, args.days)
