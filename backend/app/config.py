from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PORT_DATA_DIR") or ROOT_DIR / "data")
ARTIFACTS_DIR = Path(os.getenv("PORT_ARTIFACTS_DIR") or ROOT_DIR / "artifacts")
PROCESSED_PATH = DATA_DIR / "processed" / "port_daily.csv"
PROVENANCE_PATH = DATA_DIR / "manifests" / "provenance.json"
QUALITY_PATH = DATA_DIR / "processed" / "data_quality.json"
MODEL_PATH = ARTIFACTS_DIR / "models" / "congestion_model.joblib"
DELAY_MODEL_PATH = ARTIFACTS_DIR / "models" / "delay_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics" / "evaluation.json"
DELAY_METRICS_PATH = ARTIFACTS_DIR / "metrics" / "delay_evaluation.json"
PLOTS_DIR = ARTIFACTS_DIR / "metrics"
PORT_ID = "port-of-los-angeles"
PORT_NAME = "Port of Los Angeles (development geofence)"
FEATURES = [
    "vessel_density",
    "anchorage_vessels",
    "anchorage_dwell_hours",
    "day_of_week",
    "month",
]
TARGET = "next_day_congestion"
DELAY_TARGET = "shipment_delay"
DELAY_RISK_THRESHOLDS = {"low": 0.3, "medium": 0.6}
CORS_ORIGINS = os.getenv(
    "PORT_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
).split(",")
