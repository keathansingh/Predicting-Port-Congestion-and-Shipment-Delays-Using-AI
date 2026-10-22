from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.config import DELAY_METRICS_PATH, FEATURES, METRICS_PATH
from app.delay_training import (
    DELAY_FEATURES,
    DELAY_NUMERIC_FEATURES,
    DELAY_TARGET_DEFINITION,
)
from app.schemas import (
    DelayPredictionRequest,
    DelayPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.runtime import (
    get_dashboard_snapshot,
    get_metadata,
    get_summary,
    load_daily,
    read_json,
)
from app.services.datasets import (
    DatasetValidationError,
    get_active_dataset,
    inspect_csv,
    process_upload,
)
from app.services.vercel_dataset_storage import DatasetStorageError
from app.services import datasets

router = APIRouter(prefix="/api/v1")


@router.get("/metadata")
def metadata():
    return get_metadata()


@router.get("/summary")
def summary(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    return get_summary(start_date, end_date)


@router.get("/metrics")
def metrics():
    result = read_json(METRICS_PATH)
    if result is None:
        return {"available": False, "reason": "Model evaluation has not been run."}
    return {"available": True, **result}


@router.get("/delay/metrics")
def delay_metrics(request: Request):
    bundle = getattr(request.app.state, "delay_model_bundle", None)
    if not bundle or bundle.get("status") != "available":
        return {
            "available": False,
            "model_availability": "unavailable",
            "reason": (bundle or {}).get(
                "reason", "Shipment-delay model has not been trained."
            ),
            "training_dataset": (bundle or {}).get("training_dataset"),
        }
    result = read_json(DELAY_METRICS_PATH)
    if not result or not result.get("available"):
        return {
            "available": False,
            "model_availability": "unavailable",
            "reason": "Shipment-delay held-out evaluation artifact is unavailable.",
            "training_dataset": bundle.get("training_dataset"),
        }
    evaluation_sha = (result.get("dataset_identity") or {}).get("sha256")
    model_sha = (bundle.get("training_dataset") or {}).get("sha256")
    if evaluation_sha != model_sha:
        return {
            "available": False,
            "model_availability": "unavailable",
            "reason": "Shipment-delay model and evaluation artifacts have different dataset identities.",
            "training_dataset": bundle.get("training_dataset"),
        }
    return {
        **result,
        "model_availability": "available",
        "training_dataset": bundle.get("training_dataset"),
    }


@router.post("/delay/predict", response_model=DelayPredictionResponse)
def predict_shipment_delay(
    payload: DelayPredictionRequest, request: Request
):
    bundle = getattr(request.app.state, "delay_model_bundle", None)
    if not bundle or bundle.get("status") != "available":
        return {
            "status": "unavailable",
            "model_availability": "unavailable",
            "reason": (bundle or {}).get(
                "reason",
                "Shipment-delay model is unavailable because no valid labeled training data is loaded.",
            ),
            "evaluation_warning": None,
            "predicted_outcome": None,
            "delay_probability_percent": None,
            "risk_category": None,
            "estimated_delay_minutes": None,
            "duration_status": "unavailable",
            "probability_calibration": None,
            "probability_note": None,
            "risk_thresholds": None,
            "main_factors": [],
            "recommendations": [],
            "target_definition": DELAY_TARGET_DEFINITION,
            "dataset_provenance": (bundle or {}).get("training_dataset") or {},
            "model_version": None,
        }
    if bundle.get("features") != DELAY_FEATURES:
        raise HTTPException(
            status_code=503,
            detail="The loaded shipment-delay model feature schema is incompatible with this API.",
        )

    values = payload.model_dump()
    input_frame = pd.DataFrame(
        [{feature: values[feature] for feature in DELAY_FEATURES}],
        columns=DELAY_FEATURES,
    )
    probability = float(bundle["model"].predict_proba(input_frame)[0, 1])
    threshold = float(bundle["threshold"])
    thresholds = bundle["risk_thresholds"]
    risk = (
        "High"
        if probability >= float(thresholds["medium"])
        else "Medium"
        if probability >= float(thresholds["low"])
        else "Low"
    )

    factors = []
    for feature in DELAY_FEATURES:
        altered = input_frame.copy()
        altered.loc[0, feature] = bundle["reference_values"][feature]
        reference_probability = float(bundle["model"].predict_proba(altered)[0, 1])
        delta = probability - reference_probability
        observed = values[feature]
        factors.append(
            {
                "feature": feature,
                "observed_value": float(observed)
                if feature in DELAY_NUMERIC_FEATURES
                else str(observed),
                "reference_value": bundle["reference_values"][feature],
                "probability_change": delta,
                "direction": (
                    "increases"
                    if delta > 0.005
                    else "decreases"
                    if delta < -0.005
                    else "no_change"
                ),
                "method": (
                    "One-feature-at-a-time change from the training reference value; "
                    "not additive or causal."
                ),
            }
        )
    factors.sort(key=lambda factor: abs(factor["probability_change"]), reverse=True)

    recommendations = []
    if payload.queue_length_vessels >= payload.port_capacity_berths:
        recommendations.append(
            "Review berth allocation and queue-clearance plans; the entered queue is at least the stated berth capacity."
        )
    if payload.weather_condition.strip().lower() not in {
        "clear",
        "fair",
        "calm",
        "unknown",
    }:
        recommendations.append(
            "Check the current forecast and port operating restrictions for the reported weather condition."
        )
    max_wind = bundle.get("feature_ranges", {}).get("wind_speed_kmh", [0, 0])[1]
    if payload.wind_speed_kmh > max_wind:
        recommendations.append(
            "Verify current wind restrictions; the entered wind speed exceeds the model's observed training range."
        )
    if risk == "High":
        recommendations.append(
            "Review shipment schedules and contingency buffers against current port conditions."
        )
    elif risk == "Medium":
        recommendations.append(
            "Monitor queue, berth availability, and operating conditions before committing schedules."
        )
    if not recommendations:
        recommendations.append(
            "Continue routine monitoring; this synthetic-data estimate is not an operational guarantee."
        )
    heldout = bundle["heldout_test"]
    evaluation_warning = (
        f"Synthetic-data-only model; chronological holdout ROC-AUC was "
        f"{heldout['roc_auc']:.3f} and F1 was {heldout['f1']:.3f} "
        f"on {heldout['sample_count']} days. This does not establish real-world performance."
    )
    return {
        "status": "available",
        "model_availability": "available",
        "reason": None,
        "evaluation_warning": evaluation_warning,
        "predicted_outcome": "delayed" if probability >= threshold else "not_delayed",
        "delay_probability_percent": probability * 100,
        "risk_category": risk,
        "estimated_delay_minutes": None,
        "duration_status": "unavailable; no evaluated duration model is implemented",
        "probability_calibration": bundle["probability_calibration"],
        "probability_note": (
            "Uncalibrated delay probability estimate; it is not model confidence."
        ),
        "risk_thresholds": bundle["risk_thresholds"],
        "main_factors": factors[:5],
        "recommendations": recommendations,
        "target_definition": bundle["target_definition"],
        "dataset_provenance": bundle["training_dataset"],
        "model_version": bundle["model_version"],
    }


@router.get("/dashboard")
def dashboard(
    request: Request,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    bundle = getattr(request.app.state, "model_bundle", None)
    model_features = bundle.get("features") if bundle else None
    return get_dashboard_snapshot(start_date, end_date, model_features)


@router.get("/ports")
def ports():
    result = get_summary()
    return {
        "ports": [
            {
                "port_id": result["port_id"],
                "name": result["port_name"],
                "data_available": result["data_available"],
                "synthetic": result["synthetic"],
                "geographic_scope": get_metadata()["geographic_scope"],
            }
        ]
    }


@router.get("/ports/{port_id}/history")
def port_history(
    port_id: str,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
):
    metadata = get_metadata()
    if port_id != metadata["port_id"]:
        raise HTTPException(status_code=404, detail="Unknown port or geographic scope.")
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    frame = load_daily(start_date, end_date)
    if frame is None:
        return {"port_id": metadata["port_id"], "observations": [], "available": False}
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
    return {
        "port_id": metadata["port_id"],
        "available": True,
        "observations": records.replace({np.nan: None}).to_dict(orient="records"),
    }


async def _read_upload(file: UploadFile) -> bytes:
    contents = bytearray()
    while chunk := await file.read(min(1024 * 1024, datasets.MAX_UPLOAD_BYTES + 1 - len(contents))):
        contents.extend(chunk)
        if len(contents) > datasets.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSV uploads are limited to {datasets.MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
            )
    return bytes(contents)


def _validation_error(error: DatasetValidationError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"message": str(error), "validation": error.report},
    )


@router.get("/datasets/active")
def active_dataset():
    return get_active_dataset()


@router.post("/datasets/validate")
async def validate_dataset(file: UploadFile = File(...)):
    try:
        contents = await _read_upload(file)
        _, report = inspect_csv(contents, file.filename)
        return report
    except DatasetValidationError as error:
        raise _validation_error(error) from error
    finally:
        await file.close()


@router.post("/datasets/upload")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    source_name: str | None = Form(default=None, max_length=120),
):
    try:
        contents = await _read_upload(file)
        try:
            result = process_upload(contents, file.filename or "", source_name)
        except DatasetValidationError as error:
            raise _validation_error(error) from error
        except DatasetStorageError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise HTTPException(
                status_code=500,
                detail=f"Dataset processing failed; the active dataset was not replaced: {error}",
            ) from error
        request.app.state.data_available = True
        return result
    finally:
        await file.close()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable. Run scripts\\train_model.py first.",
        )
    model_features = bundle.get("features")
    if model_features != FEATURES:
        raise HTTPException(
            status_code=503,
            detail=(
                "The loaded model feature schema is incompatible with this API. "
                f"Expected {FEATURES}; model provides {model_features}. Retrain the model."
            ),
        )
    active_data = load_daily()
    if active_data is not None:
        missing_features = [feature for feature in model_features if feature not in active_data.columns]
        if missing_features:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The active processed dataset is incompatible with the model; missing "
                    f"features: {', '.join(missing_features)}."
                ),
            )
    values = payload.model_dump()
    frame = pd.DataFrame([[values[key] for key in FEATURES]], columns=FEATURES)
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    risk = "high" if probability >= 0.7 else "moderate" if probability >= threshold else "low"

    explanations = []
    median_values = bundle["reference_values"]
    for feature in FEATURES:
        altered = frame.copy()
        altered.loc[0, feature] = median_values[feature]
        changed_probability = float(bundle["model"].predict_proba(altered)[0, 1])
        delta = probability - changed_probability
        explanations.append(
            {
                "feature": feature,
                "observed_value": float(values[feature]),
                "reference_value": float(median_values[feature]),
                "probability_change": delta,
                "direction": (
                    "increases" if delta > 0.005 else "decreases" if delta < -0.005 else "no_change"
                ),
                "method": "one-feature-at-a-time change from training median; not additive or causal",
            }
        )
    explanations.sort(key=lambda item: abs(item["probability_change"]), reverse=True)

    reasons = []
    out_of_range = []
    for feature, observed in values.items():
        if feature not in bundle.get("feature_ranges", {}):
            continue
        lower, upper = bundle["feature_ranges"][feature]
        if observed < lower or observed > upper:
            out_of_range.append(feature)
    if bundle["training_samples"] < 50 or bundle["calibration_samples"] < 30:
        reliability = "insufficient_evidence"
        reasons.append("Training or calibration sample count is too small for a meaningful reliability assessment.")
    elif bundle.get("data_status") == "synthetic":
        reliability = "low"
        reasons.append("Training and held-out evaluation use synthetic development data only; there is no real AIS performance evidence.")
    elif out_of_range:
        reliability = "low"
        reasons.append("Input outside the observed training range: " + ", ".join(out_of_range) + ".")
    elif (
        bundle["heldout_test_samples"] >= 30
        and bundle["heldout_test_has_both_classes"]
        and bundle.get("heldout_test_roc_auc") is not None
        and bundle["heldout_test_brier_skill"] > 0
        and bundle["heldout_test_roc_auc"] >= 0.75
    ):
        reliability = "high"
        reasons.append("Input is within training ranges; held-out test evidence shows positive Brier skill and ROC-AUC of at least 0.75.")
    elif (
        bundle["heldout_test_samples"] >= 30
        and bundle["heldout_test_has_both_classes"]
        and bundle.get("heldout_test_roc_auc") is not None
        and bundle["heldout_test_brier_skill"] > 0
        and bundle["heldout_test_roc_auc"] >= 0.6
    ):
        reliability = "moderate"
        reasons.append("Input is within training ranges; held-out test evidence shows positive Brier skill and ROC-AUC of at least 0.60.")
    else:
        reliability = "low"
        reasons.append("Held-out test evidence does not meet the documented Brier-skill and discrimination thresholds.")
    reasons.append("Reliability is a separate evidence indicator, not the probability that this prediction is correct.")
    warnings = []
    if bundle.get("data_status") == "synthetic":
        warnings.append("The model was trained on synthetic development data; do not use this result operationally.")
    metadata = get_metadata()
    if metadata["evaluation_dataset_matches_active"] is False:
        warnings.append(
            "The active dataset differs from the model evaluation dataset; this model was not trained on the active upload."
        )
    if out_of_range:
        warnings.append("One or more input values fall outside the observed training range.")
    if bundle.get("target_status") == "provided":
        warnings.append(
            "The target is a supplied congestion label, not verified real-world congestion or shipment lateness."
        )
    else:
        warnings.append(
            "The target is an AIS-derived next-day congestion proxy, not verified shipment lateness."
        )
    actions = []
    if risk == "high":
        actions.append("Review scheduling buffers and monitor the port conditions more frequently.")
    elif risk == "moderate":
        actions.append("Review current port conditions before making scheduling decisions.")
    else:
        actions.append("Continue normal monitoring; this estimate does not guarantee uncongested conditions.")
    if reliability in ("low", "insufficient_evidence"):
        actions.append("Verify current conditions and input data before relying on this estimate.")

    return {
        "target": "next_day_congestion",
        "target_definition": bundle["target_definition"],
        "target_status": bundle.get("target_status"),
        "target_source": bundle.get("target_source"),
        "prediction_horizon": "Following UTC calendar day",
        "congestion_probability": probability,
        "risk_category": risk,
        "classification_threshold": threshold,
        "probability_calibration": bundle["calibration_status"],
        "prediction_reliability": reliability,
        "reliability_reasons": reasons,
        "local_explanations": explanations[:5],
        "recommended_actions": actions,
        "data_quality_warnings": warnings,
        "model_version": bundle["model_version"],
        "prediction_timestamp": datetime.now(timezone.utc),
        "shipment_delay_probability": None,
        "delay_probability": None,
        "delay_risk_category": None,
        "estimated_delay_minutes": None,
        "delay_duration_minutes": None,
        "delay_risk_explanation": None,
        "delay_explanations": None,
    }
