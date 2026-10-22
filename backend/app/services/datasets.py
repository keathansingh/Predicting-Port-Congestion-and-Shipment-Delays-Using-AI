import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.dataset_state import DATASET_LOCK
from app.config import (
    DATA_DIR,
    PORT_ID,
    PORT_NAME,
    PROCESSED_PATH,
    PROVENANCE_PATH,
    QUALITY_PATH,
)
from app.data_pipeline import (
    PORT_BOUNDS,
    REQUIRED_COLUMNS,
    build_daily_dataset,
    build_delay_dataset,
    build_unlabeled_daily_dataset,
)
from app.services.vercel_dataset_storage import persist_dataset_snapshot

MAX_UPLOAD_BYTES = 4 * 1024 * 1024 if os.getenv("VERCEL") == "1" else 25 * 1024 * 1024
MAX_UPLOAD_ROWS = 250_000
PREVIEW_ROW_COUNT = 10
_UPLOAD_LOCK = DATASET_LOCK


class DatasetValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("CSV validation failed.")
        self.report = report


def parse_csv(contents: bytes, filename: str | None) -> pd.DataFrame:
    if not filename or Path(filename).suffix.lower() != ".csv":
        raise DatasetValidationError(
            {
                "valid": False,
                "errors": ["Choose a file with a .csv extension."],
                "warnings": [],
                "input_rows": 0,
                "columns": [],
                "missing_columns": [],
                "missing_values": {},
                "invalid_dates": 0,
                "invalid_numeric_values": {},
                "invalid_mmsi_values": 0,
                "duplicate_records": 0,
                "preview": [],
            }
        )
    if not contents:
        raise DatasetValidationError(
            {
                "valid": False,
                "errors": ["The uploaded file is empty."],
                "warnings": [],
                "input_rows": 0,
                "columns": [],
                "missing_columns": [],
                "missing_values": {},
                "invalid_dates": 0,
                "invalid_numeric_values": {},
                "invalid_mmsi_values": 0,
                "duplicate_records": 0,
                "preview": [],
            }
        )
    try:
        frame = pd.read_csv(
            BytesIO(contents),
            dtype={"MMSI": "string"},
            encoding="utf-8-sig",
            low_memory=False,
            nrows=MAX_UPLOAD_ROWS + 1,
            on_bad_lines="error",
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise DatasetValidationError(
            {
                "valid": False,
                "errors": [f"Could not parse this CSV: {error}"],
                "warnings": [],
                "input_rows": 0,
                "columns": [],
                "missing_columns": [],
                "missing_values": {},
                "invalid_dates": 0,
                "invalid_numeric_values": {},
                "invalid_mmsi_values": 0,
                "duplicate_records": 0,
                "preview": [],
            }
        ) from error
    if len(frame) > MAX_UPLOAD_ROWS:
        raise DatasetValidationError(
            {
                "valid": False,
                "errors": [f"Files may contain at most {MAX_UPLOAD_ROWS:,} rows."],
                "warnings": [],
                "input_rows": len(frame),
                "columns": list(frame.columns),
                "missing_columns": [column for column in REQUIRED_COLUMNS if column not in frame.columns],
                "missing_values": {},
                "invalid_dates": 0,
                "invalid_numeric_values": {},
                "invalid_mmsi_values": 0,
                "duplicate_records": 0,
                "preview": [],
            }
        )
    return frame


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    errors: list[str] = []
    warnings: list[str] = []
    missing_values: dict[str, int] = {}
    invalid_numeric_values: dict[str, int] = {}
    invalid_dates = 0
    invalid_mmsi_values = 0
    duplicate_records = 0

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    elif frame.empty:
        errors.append("The CSV has headers but no data rows.")
    else:
        for column in REQUIRED_COLUMNS:
            values = frame[column]
            if column == "MMSI":
                normalized = values.astype("string").str.strip()
                missing = values.isna() | normalized.eq("").fillna(True)
                valid_mmsi = normalized.str.fullmatch(r"\d{9}", na=False)
                invalid_mmsi_values = int((~missing & ~valid_mmsi).sum())
                if invalid_mmsi_values:
                    errors.append(
                        f"MMSI has {invalid_mmsi_values:,} invalid value(s); identifiers must contain nine digits."
                    )
            else:
                missing = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
            count = int(missing.sum())
            if count:
                missing_values[column] = count
                errors.append(f"{column} has {count:,} missing value(s).")

        timestamps = pd.to_datetime(frame["BaseDateTime"], errors="coerce", utc=True, format="mixed")
        invalid_dates = int(timestamps.isna().sum())
        if invalid_dates:
            errors.append(f"BaseDateTime has {invalid_dates:,} invalid date/time value(s).")

        for column, lower, upper in (
            ("LAT", -90.0, 90.0),
            ("LON", -180.0, 180.0),
            ("SOG", 0.0, 102.3),
        ):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            valid = numeric.notna() & numeric.map(math.isfinite)
            if column == "LAT":
                valid &= numeric.between(lower, upper)
            elif column == "LON":
                valid &= numeric.between(lower, upper)
            else:
                valid &= numeric.between(lower, upper)
            invalid_numeric_values[column] = int((~valid).sum())
            if invalid_numeric_values[column]:
                errors.append(
                    f"{column} has {invalid_numeric_values[column]:,} non-numeric or out-of-range value(s)."
                )

        normalized_mmsi = frame["MMSI"].astype("string").str.strip()
        duplicate_records = int(
            pd.DataFrame({"MMSI": normalized_mmsi, "timestamp": timestamps})
            .dropna()
            .duplicated(["MMSI", "timestamp"])
            .sum()
        )
        if duplicate_records:
            warnings.append(
                f"{duplicate_records:,} duplicate MMSI/timestamp record(s) will be removed during processing."
            )

    preview: list[dict[str, Any]] = []
    for record in frame.head(PREVIEW_ROW_COUNT).to_dict(orient="records"):
        preview.append(
            {
                str(key): None if pd.isna(value) else str(value)
                for key, value in record.items()
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "input_rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "invalid_dates": invalid_dates,
        "invalid_numeric_values": invalid_numeric_values,
        "invalid_mmsi_values": invalid_mmsi_values,
        "duplicate_records": duplicate_records,
        "preview": preview,
    }


def inspect_csv(contents: bytes, filename: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = parse_csv(contents, filename)
    report = validate_frame(frame)
    if not report["valid"]:
        raise DatasetValidationError(report)
    return frame, report


def _geographic_scope(frame: pd.DataFrame) -> dict[str, Any]:
    latitude = pd.to_numeric(frame["LAT"], errors="coerce")
    longitude = pd.to_numeric(frame["LON"], errors="coerce")
    los_angeles = (
        latitude.between(PORT_BOUNDS["lat_min"], PORT_BOUNDS["lat_max"])
        & longitude.between(PORT_BOUNDS["lon_min"], PORT_BOUNDS["lon_max"])
    ).all()
    coverage = {
        "latitude": {"min": float(latitude.min()), "max": float(latitude.max())},
        "longitude": {"min": float(longitude.min()), "max": float(longitude.max())},
    }
    if los_angeles:
        return {
            "port_id": PORT_ID,
            "label": (
                "Configured Port of Los Angeles development geofence "
                "(coordinate match; port identity not independently verified)"
            ),
            "identified": True,
            "identification_method": (
                "All uploaded coordinates fall within the configured Port of Los Angeles "
                "development geofence. This confirms a coordinate match only; it does not "
                "independently verify port identity."
            ),
            "coordinate_coverage": coverage,
            "geofence": PORT_BOUNDS,
        }
    return {
        "port_id": "uploaded-ais-area",
        "label": "Uploaded AIS dataset — geographic area not verified",
        "identified": False,
        "identification_method": (
            "Coordinate extent is reported; no reliable named-port mapping is configured."
        ),
        "coordinate_coverage": coverage,
        "geofence": None,
    }


def display_geographic_scope(provenance: dict[str, Any]) -> dict[str, Any] | None:
    scope = provenance.get("geographic_scope")
    if not scope:
        return None
    if (
        provenance.get("data_status") == "uploaded"
        and scope.get("port_id") == PORT_ID
        and scope.get("identified")
    ):
        scope = dict(scope)
        scope["label"] = (
            "Configured Port of Los Angeles development geofence "
            "(coordinate match; port identity not independently verified)"
        )
        scope["identification_method"] = (
            "All uploaded coordinates fall within the configured Port of Los Angeles "
            "development geofence. This confirms a coordinate match only; it does not "
            "independently verify port identity."
        )
    return scope


def _replace_dataset_files(staged: dict[Path, Path]) -> None:
    replaced: list[tuple[Path, Path | None]] = []
    try:
        for source, destination in staged.items():
            backup = source.parent / f"backup-{len(replaced)}"
            previous = None
            if destination.exists():
                shutil.copy2(destination, backup)
                previous = backup
            os.replace(source, destination)
            replaced.append((destination, previous))
    except OSError:
        for destination, backup in reversed(replaced):
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        raise


def process_upload(
    contents: bytes,
    filename: str,
    source_name: str | None = None,
) -> dict[str, Any]:
    frame, report = inspect_csv(contents, filename)
    geographic_scope = _geographic_scope(frame)
    if geographic_scope["identified"]:
        if "SyntheticDelayLabel" in frame.columns or "SyntheticDelayMinutes" in frame.columns:
            daily, quality = build_delay_dataset(frame)
        else:
            daily, quality = build_daily_dataset(frame)
        if "SyntheticCongestionLabel" in frame.columns:
            if quality.get("target_status") == "provided":
                quality["target_source"] = quality.get(
                    "target_source",
                    "SyntheticCongestionLabel column in the source dataset; aggregated by UTC day using max value.",
                )
                quality["target_definition"] = quality.get(
                    "target_definition",
                    "The uploaded synthetic congestion label is used as the binary target when it is supplied and valid.",
                )
                quality["provided_target_columns_used"] = ["SyntheticCongestionLabel"]
                quality["provided_target_columns_ignored"] = []
            else:
                quality["target_status"] = "derived_proxy"
                quality["target_source"] = (
                    "Derived from AIS messages using the configured Port of Los Angeles "
                    "candidate-anchorage geofence."
                )
                quality["target_definition"] = quality.get("label_definition") or quality.get("target_definition")
                quality["provided_target_columns_used"] = []
                quality["provided_target_columns_ignored"] = ["SyntheticCongestionLabel"]
        else:
            quality["target_status"] = "derived_proxy"
            quality["target_source"] = (
                "Derived from AIS messages using the configured Port of Los Angeles "
                "candidate-anchorage geofence."
            )
            quality["target_definition"] = quality["label_definition"]
            quality["provided_target_columns_ignored"] = []
    else:
        daily, quality = build_unlabeled_daily_dataset(frame)
    quality["coordinate_coverage"] = geographic_scope["coordinate_coverage"]
    if daily.empty:
        raise DatasetValidationError(
            {
                **report,
                "valid": False,
                "errors": [
                    "No daily AIS observations could be built from the valid uploaded rows."
                ],
            }
        )

    digest = hashlib.sha256(contents).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    synthetic = False
    if "SyntheticData" in frame.columns:
        synthetic_markers = (
            frame["SyntheticData"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "synthetic"})
        )
        synthetic = bool(synthetic_markers.all())
    active_source = source_name.strip() if source_name and source_name.strip() else "User-uploaded AIS CSV"
    if synthetic and "synthetic" not in active_source.lower():
        active_source = f"Synthetic development data — {active_source}"
    quality.update(
        {
            "source_dataset_sha256": digest,
            "processed_rows": int(len(daily)),
            "observation_start": daily["date"].min().date().isoformat(),
            "observation_end": daily["date"].max().date().isoformat(),
            "processed_at": now,
            "validation_report": {key: value for key, value in report.items() if key != "preview"},
        }
    )
    previous_provenance: dict[str, Any] = {}
    if PROVENANCE_PATH.exists():
        with PROVENANCE_PATH.open(encoding="utf-8") as handle:
            previous_provenance = json.load(handle)
    provenance = {
        "data_status": "synthetic" if synthetic else "uploaded",
        "source_name": active_source,
        "source_url": None,
        "acquisition_date": datetime.now(timezone.utc).date().isoformat(),
        "filename": Path(filename).name,
        "file_size_bytes": len(contents),
        "sha256": digest,
        "rows": int(len(frame)),
        "original_columns": [str(column) for column in frame.columns],
        "columns": [str(column) for column in frame.columns],
        "synthetic": synthetic,
        "synthetic_notice": (
            "Generated observations; not real vessel broadcasts or public source data."
            if synthetic
            else None
        ),
        "processed_rows": int(len(daily)),
        "processed_observation_start": quality["observation_start"],
        "processed_observation_end": quality["observation_end"],
        "delay_target_status": quality.get("delay_target_status"),
        "delay_target_source": quality.get("delay_target_source"),
        "delay_target_distribution": quality.get("delay_target_distribution"),
        "processed_at": now,
        "quality_report": str(QUALITY_PATH.relative_to(DATA_DIR.parent)),
        "uploaded_csv": str(Path("uploads") / f"{digest}.csv"),
        "previous_dataset": {
            key: previous_provenance[key]
            for key in ("data_status", "source_name", "filename", "sha256")
            if key in previous_provenance
        },
        "model_retrained": False,
        "model_note": "Dataset activation does not immediately retrain or replace running models; restart the API to load models for this dataset.",
        "geographic_scope": geographic_scope,
        "target_status": quality["target_status"],
        "target_source": quality["target_source"],
        "target_definition": quality["target_definition"],
    }
    upload_directory = DATA_DIR / "uploads"
    upload_path = upload_directory / f"{digest}.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    upload_directory.mkdir(parents=True, exist_ok=True)
    with _UPLOAD_LOCK, tempfile.TemporaryDirectory(prefix=".dataset-upload-", dir=DATA_DIR) as temp_name:
        staging = Path(temp_name)
        daily_path = staging / "port_daily.csv"
        quality_path = staging / "data_quality.json"
        provenance_path = staging / "provenance.json"
        raw_path = staging / "uploaded.csv"
        daily.to_csv(daily_path, index=False)
        quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        raw_path.write_bytes(contents)
        persist_dataset_snapshot(
            daily_path,
            quality_path,
            provenance_path,
            raw_path,
        )
        _replace_dataset_files(
            {
                raw_path: upload_path,
                daily_path: PROCESSED_PATH,
                quality_path: QUALITY_PATH,
                provenance_path: PROVENANCE_PATH,
            }
        )

    return {
        "status": "processed",
        "filename": Path(filename).name,
        "source_name": active_source,
        "input_rows": int(len(frame)),
        "processed_rows": int(len(daily)),
        "date_coverage": {
            "start": quality["observation_start"],
            "end": quality["observation_end"],
        },
        "last_processed_at": now,
        "sha256": digest,
        "validation": {key: value for key, value in report.items() if key != "preview"},
        "model_retrained": False,
        "model_note": provenance["model_note"],
        "geographic_scope": geographic_scope,
        "target_status": quality["target_status"],
        "target_source": quality["target_source"],
    }


def get_active_dataset() -> dict[str, Any]:
    with DATASET_LOCK:
        if not PROVENANCE_PATH.exists():
            return {"available": False}
        with PROVENANCE_PATH.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        quality: dict[str, Any] = {}
        if QUALITY_PATH.exists():
            with QUALITY_PATH.open(encoding="utf-8") as handle:
                quality = json.load(handle)
        daily = (
            pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
            if PROCESSED_PATH.is_file()
            else None
        )
        date_coverage = (
            {
                "start": daily["date"].min().date().isoformat(),
                "end": daily["date"].max().date().isoformat(),
            }
            if daily is not None and not daily.empty
            else None
        )
        return {
            "available": PROCESSED_PATH.is_file(),
            "processing_status": "processed" if PROCESSED_PATH.is_file() else "unavailable",
            "source_name": provenance.get("source_name"),
            "source_url": provenance.get("source_url"),
            "filename": provenance.get("filename"),
            "data_status": provenance.get("data_status", "not_processed"),
            "synthetic": provenance.get("data_status") == "synthetic",
            "input_rows": provenance.get("rows"),
            "processed_rows": len(daily) if daily is not None else 0,
            "date_coverage": date_coverage,
            "last_processed_at": provenance.get("processed_at"),
            "sha256": provenance.get("sha256"),
            "model_retrained": provenance.get("model_retrained", None),
            "model_note": provenance.get("model_note"),
            "geographic_scope": display_geographic_scope(provenance),
            "target_status": quality.get("target_status"),
            "target_source": quality.get("target_source"),
            "quality": quality,
        }
