import json
from datetime import date
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import (
    DATA_DIR,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    PORT_ID,
    PORT_NAME,
    PROCESSED_PATH,
    PROVENANCE_PATH,
    QUALITY_PATH,
)
from app.data_pipeline import PORT_BOUNDS
from app.delay_training import load_delay_model
from app.services.dataset_state import DATASET_LOCK
from app.services.datasets import display_geographic_scope


def read_json(path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


def load_daily(
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame | None:
    with DATASET_LOCK:
        if not PROCESSED_PATH.is_file():
            return None
        frame = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
        if start_date is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date is not None:
            frame = frame.loc[frame["date"] < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
        return frame


def _dataset_match(active: dict[str, Any], evaluation: dict[str, Any]) -> bool | None:
    if active.get("sha256") and evaluation.get("sha256"):
        return active["sha256"] == evaluation["sha256"]
    if (
        active.get("data_status")
        and evaluation.get("data_status")
        and active["data_status"] != evaluation["data_status"]
    ):
        return False
    return None


def _model_training_dataset(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return bundle.get("training_dataset") or {
        "data_status": bundle.get("data_status"),
        "source_name": None,
        "filename": None,
        "sha256": None,
        "identity_recorded": False,
    }


def _geographic_scope(provenance: dict[str, Any]) -> dict[str, Any]:
    scope = display_geographic_scope(provenance)
    if scope:
        return scope
    if provenance.get("data_status") == "uploaded":
        return {
            "port_id": "uploaded-ais-area",
            "label": "Uploaded AIS dataset — geographic area not verified",
            "identified": False,
            "identification_method": (
                "Coordinate extent is unavailable; no reliable named-port mapping is configured."
            ),
            "coordinate_coverage": None,
            "geofence": None,
        }
    return {
        "port_id": PORT_ID,
        "label": PORT_NAME,
        "identified": True,
        "identification_method": (
            "The processed dataset uses the configured Port of Los Angeles development geofence."
        ),
        "coordinate_coverage": None,
        "geofence": PORT_BOUNDS,
    }


def _metadata(
    provenance: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    daily_data: pd.DataFrame | None,
    model_features: list[str] | None,
    model_training_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_dataset = metrics.get("training_dataset") or {
        "data_status": metrics.get("data_status"),
        "source_name": None,
        "filename": None,
        "sha256": None,
        "identity_recorded": False,
    }
    active_dataset = {
        "data_status": provenance.get("data_status"),
        "sha256": provenance.get("sha256"),
    }
    expected_features = model_features or metrics.get("features", FEATURES)
    geographic_scope = _geographic_scope(provenance)
    target_definition = (
        quality.get("target_definition")
        or quality.get("label_definition")
        or metrics.get(
            "target_definition",
            "Next UTC day has at least 25 distinct AIS-observed vessels in a candidate-anchorage geofence.",
        )
    )
    target_status = quality.get("target_status") or (
        "derived_proxy" if quality.get("label_definition") else "available"
    )
    columns = list(daily_data.columns) if daily_data is not None else []
    missing_features = [feature for feature in expected_features if feature not in columns]
    incompatible_model_features = model_features is not None and model_features != FEATURES
    incompatibility_reasons = []
    if incompatible_model_features:
        incompatibility_reasons.append(
            f"Loaded model expects {model_features}; API expects {FEATURES}"
        )
    if missing_features:
        incompatibility_reasons.append(
            "Active processed dataset is missing model features: "
            + ", ".join(missing_features)
        )
    if not geographic_scope["identified"]:
        incompatibility_reasons.append(
            "The active dataset has no verified Port of Los Angeles scope; "
            "the loaded model's geographic training scope is not compatible."
        )
    observation_start = (
        daily_data["date"].min().date().isoformat()
        if daily_data is not None and not daily_data.empty
        else None
    )
    observation_end = (
        daily_data["date"].max().date().isoformat()
        if daily_data is not None and not daily_data.empty
        else None
    )
    return {
        "project_name": "Port Congestion and Shipment Delay Prediction System",
        "port_id": geographic_scope["port_id"],
        "port_name": geographic_scope["label"],
        "target": metrics.get("target", "next_day_congestion"),
        "target_definition": target_definition,
        "target_rule": quality.get("candidate_anchorage_rule"),
        "target_status": target_status,
        "target_source": quality.get("target_source"),
        "geographic_scope": geographic_scope,
        "prediction_horizon": "Following UTC calendar day",
        "data_status": provenance.get("data_status", "not_processed"),
        "synthetic": provenance.get("data_status") == "synthetic",
        "source_url": provenance.get("source_url"),
        "source_name": provenance.get("source_name"),
        "acquisition_date": provenance.get("acquisition_date"),
        "observation_start": observation_start,
        "observation_end": observation_end,
        "processed_rows": len(daily_data) if daily_data is not None else 0,
        "last_processed_at": provenance.get("processed_at"),
        "model_version": metrics.get("model_version"),
        "dataset_sha256": provenance.get("sha256"),
        "model_training_dataset": model_training_dataset,
        "evaluation_dataset": evaluation_dataset,
        "evaluation_dataset_matches_active": _dataset_match(
            active_dataset, evaluation_dataset
        ),
        "prediction_compatible": (
            not incompatible_model_features
            and not missing_features
            and geographic_scope["identified"]
            if columns and model_features is not None
            else None
        ),
        "prediction_incompatibility": "; ".join(incompatibility_reasons) or None,
        "limitations": [
            "This is an AIS-derived congestion proxy, not a verified shipment-delay prediction.",
            "Synthetic development data is not evidence of real maritime performance.",
            "The geographic and anchorage rules are coarse and require port-specific validation.",
        ],
    }


def get_metadata() -> dict[str, Any]:
    with DATASET_LOCK:
        provenance = read_json(PROVENANCE_PATH, {})
        metrics = read_json(METRICS_PATH, {})
        quality = read_json(QUALITY_PATH, {})
        bundle = load_bundle()
        model_features = bundle.get("features") if bundle else None
        daily_data = (
            pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
            if PROCESSED_PATH.is_file()
            else None
        )
        return _metadata(
            provenance,
            metrics,
            quality,
            daily_data,
            model_features,
            _model_training_dataset(bundle),
        )


def get_summary(
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    with DATASET_LOCK:
        all_rows = load_daily()
        frame = load_daily(start_date, end_date)
        quality = read_json(QUALITY_PATH, {})
        metadata = get_metadata()
        return _summary(frame, all_rows, quality, metadata)


def _summary(
    frame: pd.DataFrame | None,
    all_rows: pd.DataFrame | None,
    quality: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    scope_id = metadata["port_id"]
    scope_name = metadata["port_name"]
    if frame is None or frame.empty:
        return {
            "data_available": False,
            "dataset_available": all_rows is not None and not all_rows.empty,
            "port_id": scope_id,
            "port_name": scope_name,
            "observations": 0,
            "target_observations": 0,
            "target_available": False,
            "congestion_days": None,
            "congestion_rate": None,
            "coverage": None,
            "synthetic": metadata["synthetic"],
            "target_status": metadata["target_status"],
            "target_definition": metadata["target_definition"],
            "quality": quality,
        }
    labels = (
        pd.to_numeric(frame["next_day_congestion"], errors="coerce")
        if "next_day_congestion" in frame
        else pd.Series(index=frame.index, dtype="float64")
    )
    labels = labels.where(labels.isin([0, 1])).dropna()
    target_observations = len(labels)
    return {
        "data_available": True,
        "dataset_available": True,
        "port_id": scope_id,
        "port_name": scope_name,
        "observations": len(frame),
        "target_observations": target_observations,
        "target_available": target_observations > 0,
        "congestion_days": int(labels.sum()) if target_observations else None,
        "congestion_rate": float(labels.mean()) if target_observations else None,
        "coverage": {
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
        },
        "synthetic": metadata["synthetic"],
        "target_status": metadata["target_status"],
        "target_definition": metadata["target_definition"],
        "quality": quality,
    }


def get_dashboard_snapshot(
    start_date: date | None = None,
    end_date: date | None = None,
    model_features: list[str] | None = None,
) -> dict[str, Any]:
    with DATASET_LOCK:
        all_rows = (
            pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
            if PROCESSED_PATH.is_file()
            else None
        )
        frame = all_rows
        if frame is not None and start_date is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start_date, tz="UTC")]
        if frame is not None and end_date is not None:
            frame = frame.loc[
                frame["date"] < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            ]
        provenance = read_json(PROVENANCE_PATH, {})
        quality = read_json(QUALITY_PATH, {})
        metrics = read_json(METRICS_PATH, {})
        bundle = load_bundle()
        metadata = _metadata(
            provenance,
            metrics,
            quality,
            all_rows,
            model_features,
            _model_training_dataset(bundle),
        )
        summary = _summary(frame, all_rows, quality, metadata)
        active_coverage = (
            {
                "start": all_rows["date"].min().date().isoformat(),
                "end": all_rows["date"].max().date().isoformat(),
            }
            if all_rows is not None and not all_rows.empty
            else None
        )
        active_dataset = {
            "available": PROCESSED_PATH.is_file(),
            "processing_status": "processed" if PROCESSED_PATH.is_file() else "unavailable",
            "source_name": provenance.get("source_name"),
            "filename": provenance.get("filename"),
            "data_status": provenance.get("data_status", "not_processed"),
            "synthetic": provenance.get("data_status") == "synthetic",
            "input_rows": provenance.get("rows"),
            "processed_rows": len(all_rows) if all_rows is not None else 0,
            "date_coverage": active_coverage,
            "last_processed_at": provenance.get("processed_at"),
            "sha256": provenance.get("sha256"),
            "model_retrained": provenance.get("model_retrained"),
            "model_note": provenance.get("model_note"),
            "geographic_scope": metadata["geographic_scope"],
            "target_status": metadata["target_status"],
            "target_source": metadata["target_source"],
        }
        history = []
        if frame is not None:
            fields = [
                "date",
                "vessel_density",
                "anchorage_vessels",
                "anchorage_dwell_hours",
                "ais_congestion_proxy",
                "synthetic_congestion_label",
                "next_day_congestion",
            ]
            records = frame[[field for field in fields if field in frame.columns]].copy()
            records["date"] = records["date"].dt.strftime("%Y-%m-%d")
            records = records.astype(object).where(pd.notna(records), None)
            history = records.to_dict(orient="records")
        dashboard_metrics = dict(metrics or {})
        dashboard_metrics.setdefault("training_dataset", metadata["evaluation_dataset"])
        return {
            "metadata": metadata,
            "summary": summary,
            "metrics": {"available": metrics is not None, **dashboard_metrics},
            "ports": [
                {
                    "port_id": metadata["port_id"],
                    "name": metadata["port_name"],
                    "data_available": summary["data_available"],
                    "synthetic": summary["synthetic"],
                    "geographic_scope": metadata["geographic_scope"],
                }
            ],
            "history": history,
            "active_dataset": active_dataset,
        }
