import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    DELAY_METRICS_PATH,
    DELAY_MODEL_PATH,
    DELAY_RISK_THRESHOLDS,
    PROCESSED_PATH,
    PROVENANCE_PATH,
)

DELAY_NUMERIC_FEATURES = [
    "vessel_arrivals_daily",
    "queue_length_vessels",
    "port_capacity_berths",
    "wind_speed_kmh",
    "day_of_week",
    "month",
]
DELAY_CATEGORICAL_FEATURES = ["weather_condition"]
DELAY_FEATURES = DELAY_NUMERIC_FEATURES + DELAY_CATEGORICAL_FEATURES
DELAY_ARTIFACT_VERSION = "shipment-delay-v3"
DELAY_TARGET_DEFINITION = (
    "SyntheticDelayLabel aggregated by UTC target day using max; operational "
    "features are measured on the preceding UTC day."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_is_synthetic(provenance: dict[str, Any]) -> bool:
    if provenance.get("synthetic") is True:
        return True
    source = str(provenance.get("source_name") or "").lower()
    return "synthetic" in source or "syntheticdelaylabel" in str(
        provenance.get("target_source") or ""
    ).lower()


def _unavailable(reason: str, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "artifact_version": DELAY_ARTIFACT_VERSION,
        "target": "SyntheticDelayLabel",
        "target_definition": DELAY_TARGET_DEFINITION,
        "training_dataset": {
            "data_status": "synthetic" if _dataset_is_synthetic(provenance) else provenance.get("data_status"),
            "source_name": provenance.get("source_name"),
            "filename": provenance.get("filename"),
            "sha256": provenance.get("sha256"),
            "identity_recorded": bool(provenance.get("sha256")),
        },
    }


def _archive_previous_metrics(current_sha256: str | None) -> None:
    if not DELAY_METRICS_PATH.is_file():
        return
    previous = _read_json(DELAY_METRICS_PATH)
    previous_identity = (previous.get("training_dataset") or {}).get("sha256")
    if previous_identity == current_sha256:
        return
    suffix = previous_identity or "identity-unrecorded"
    archive = DELAY_METRICS_PATH.with_name(f"delay_evaluation_{suffix}.json")
    if not archive.exists():
        shutil.copy2(DELAY_METRICS_PATH, archive)


def _period(dates: pd.Series) -> dict[str, str]:
    normalized = pd.to_datetime(dates, errors="coerce", utc=True).dropna()
    return {
        "start": normalized.min().date().isoformat(),
        "end": normalized.max().date().isoformat(),
    }


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, DELAY_NUMERIC_FEATURES),
            ("categorical", categorical, DELAY_CATEGORICAL_FEATURES),
        ]
    )


def train_delay_model(
    frame: pd.DataFrame,
    provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = provenance or {}
    missing = [column for column in ["date", "shipment_delay", *DELAY_FEATURES] if column not in frame]
    if missing:
        unavailable = _unavailable(
            "Delay training data is missing required labeled columns: "
            + ", ".join(missing)
            + ". Reprocess a dataset containing SyntheticDelayLabel.",
            provenance,
        )
        return unavailable, {"available": False, "reason": unavailable["reason"]}

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    targets = pd.to_numeric(data["shipment_delay"], errors="coerce")
    invalid_target = data["shipment_delay"].notna() & (
        targets.isna() | ~targets.isin([0, 1])
    )
    if invalid_target.any():
        unavailable = _unavailable(
            "Processed SyntheticDelayLabel values are not binary 0/1; refusing to train.",
            provenance,
        )
        return unavailable, {"available": False, "reason": unavailable["reason"]}
    data["shipment_delay"] = targets
    data = data.dropna(subset=["date", "shipment_delay"]).sort_values("date").reset_index(drop=True)
    data["shipment_delay"] = data["shipment_delay"].astype(int)
    distribution = {
        "not_delayed": int((data["shipment_delay"] == 0).sum()),
        "delayed": int((data["shipment_delay"] == 1).sum()),
    }
    if len(data) < 50 or data["shipment_delay"].nunique() != 2:
        unavailable = _unavailable(
            "Delay training requires at least 50 valid labeled daily examples with both classes; "
            f"found {len(data)} examples ({distribution['not_delayed']} not delayed, "
            f"{distribution['delayed']} delayed).",
            provenance,
        )
        unavailable["target_distribution"] = distribution
        return unavailable, {"available": False, "reason": unavailable["reason"]}

    split_index = int(len(data) * 0.8)
    train, test = data.iloc[:split_index], data.iloc[split_index:]
    if train["shipment_delay"].nunique() != 2 or test["shipment_delay"].nunique() != 2:
        unavailable = _unavailable(
            "The chronological 80/20 split must contain both target classes in train and test; "
            "the available labels do not support held-out evaluation.",
            provenance,
        )
        unavailable["target_distribution"] = distribution
        return unavailable, {"available": False, "reason": unavailable["reason"]}

    model = Pipeline(
        [
            ("preprocessor", _preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    X_train = train[DELAY_FEATURES].copy()
    X_test = test[DELAY_FEATURES].copy()
    for column in DELAY_CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].fillna("Unknown").astype(str)
        X_test[column] = X_test[column].fillna("Unknown").astype(str)
    y_train = train["shipment_delay"].to_numpy()
    y_test = test["shipment_delay"].to_numpy()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "available": True,
        "model": "balanced_logistic_regression",
        "target": "SyntheticDelayLabel",
        "target_definition": DELAY_TARGET_DEFINITION,
        "probability_calibration": "uncalibrated",
        "probability_calibration_note": (
            "Probabilities are uncalibrated estimates from a class-weighted classifier; "
            "they are not model confidence."
        ),
        "dataset_identity": {
            "data_status": "synthetic" if _dataset_is_synthetic(provenance) else provenance.get("data_status"),
            "source_name": provenance.get("source_name"),
            "filename": provenance.get("filename"),
            "sha256": provenance.get("sha256"),
        },
        "target_distribution": distribution,
        "features": DELAY_FEATURES,
        "split_strategy": "chronological holdout; earliest 80% train, latest 20% final test",
        "train": {
            "sample_count": len(train),
            "period": _period(train["date"]),
            "class_distribution": {
                "not_delayed": int((y_train == 0).sum()),
                "delayed": int((y_train == 1).sum()),
            },
        },
        "test": {
            "sample_count": len(test),
            "period": _period(test["date"]),
            "class_distribution": {
                "not_delayed": int((y_test == 0).sum()),
                "delayed": int((y_test == 1).sum()),
            },
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        },
        "duration_regression": {
            "available": False,
            "reason": "No independently validated duration model is implemented.",
        },
    }
    numeric_reference = train[DELAY_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    reference_values: dict[str, float | str] = {}
    feature_ranges: dict[str, list[float]] = {}
    for column in DELAY_NUMERIC_FEATURES:
        values = numeric_reference[column].dropna()
        reference_values[column] = float(values.median()) if not values.empty else 0.0
        feature_ranges[column] = (
            [float(values.min()), float(values.max())] if not values.empty else [0.0, 0.0]
        )
    reference_values["weather_condition"] = str(
        train["weather_condition"].fillna("Unknown").astype(str).mode().iloc[0]
    )
    bundle = {
        "status": "available",
        "artifact_version": DELAY_ARTIFACT_VERSION,
        "model": model,
        "features": DELAY_FEATURES,
        "numeric_features": DELAY_NUMERIC_FEATURES,
        "categorical_features": DELAY_CATEGORICAL_FEATURES,
        "threshold": 0.5,
        "risk_thresholds": DELAY_RISK_THRESHOLDS,
        "probability_calibration": "uncalibrated",
        "model_version": DELAY_ARTIFACT_VERSION,
        "target": "SyntheticDelayLabel",
        "target_definition": DELAY_TARGET_DEFINITION,
        "target_distribution": distribution,
        "heldout_test": {
            "sample_count": metrics["test"]["sample_count"],
            "roc_auc": metrics["test"]["roc_auc"],
            "f1": metrics["test"]["f1"],
        },
        "reference_values": reference_values,
        "feature_ranges": feature_ranges,
        "training_dataset": {
            **metrics["dataset_identity"],
            "identity_recorded": bool(provenance.get("sha256")),
        },
    }
    return bundle, metrics


def load_delay_model() -> dict[str, Any]:
    provenance = _read_json(PROVENANCE_PATH)
    if not PROCESSED_PATH.is_file():
        return _unavailable(
            "Shipment-delay model unavailable: no processed dataset is available.", provenance
        )
    try:
        frame = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    except (OSError, pd.errors.ParserError, ValueError) as error:
        return _unavailable(f"Shipment-delay training data could not be read: {error}", provenance)

    if DELAY_MODEL_PATH.is_file():
        try:
            cached = joblib.load(DELAY_MODEL_PATH)
            if not isinstance(cached, dict):
                cached = {}
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            ImportError,
            AttributeError,
            pickle.UnpicklingError,
        ):
            cached = {}
        if (
            cached.get("artifact_version") == DELAY_ARTIFACT_VERSION
            and cached.get("training_dataset", {}).get("sha256") == provenance.get("sha256")
        ):
            return cached

    if os.getenv("VERCEL") == "1":
        return _unavailable(
            "The active dataset differs from the bundled shipment-delay model. "
            "Retrain and redeploy the model artifact for this dataset.",
            provenance,
        )

    bundle, metrics = train_delay_model(frame, provenance)
    if bundle["status"] == "available":
        _archive_previous_metrics(provenance.get("sha256"))
        DELAY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, DELAY_MODEL_PATH)
        DELAY_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DELAY_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle
