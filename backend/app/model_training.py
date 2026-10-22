import json
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
from sklearn.base import clone

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from app.config import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    PROCESSED_PATH,
    TARGET,
)
from app.data_pipeline import TARGET_DEFINITION

SEED = 42
THRESHOLD = 0.5


def _metrics(y_true, probabilities, threshold: float = THRESHOLD) -> dict:
    labels = (probabilities >= threshold).astype(int)
    result = {
        "sample_count": int(len(y_true)),
        "classification_threshold": float(threshold),
        "class_distribution": {
            "negative": int((np.asarray(y_true) == 0).sum()),
            "positive": int((np.asarray(y_true) == 1).sum()),
        },
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": None,
        "roc_auc_reason": None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc_reason"] = "ROC-AUC requires both classes in the evaluation sample."
    return result


def _select_f1_threshold(y_true, probabilities) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return {
            "threshold": THRESHOLD,
            "precision": float(precision[-1]),
            "recall": float(recall[-1]),
            "f1": 0.0,
            "predicted_positive": int((probabilities >= THRESHOLD).sum()),
        }
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_indices = np.flatnonzero(np.isclose(f1_scores, f1_scores.max()))
    best_index = int(best_indices[0])
    threshold = float(thresholds[best_index])
    return {
        "threshold": threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
        "predicted_positive": int((probabilities >= threshold).sum()),
    }


def _calibration_points(y_true, probabilities) -> list[dict] | None:
    if len(np.unique(y_true)) != 2:
        return None
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=5, strategy="quantile"
    )
    return [
        {
            "mean_predicted_probability": float(mean_probability),
            "observed_event_fraction": float(event_fraction),
        }
        for event_fraction, mean_probability in zip(observed, predicted)
    ]


def _save_plots(test_y, selected_probabilities, importance):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(list(importance), list(importance.values()), color="#2c91a5")
    axis.set_title("Global feature importance")
    axis.set_ylabel("Normalized model importance")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "feature_importance.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot([0, 1], [0, 1], "--", color="#6c7888", label="Perfect calibration")
    points = _calibration_points(test_y, selected_probabilities)
    if points:
        axis.plot(
            [point["mean_predicted_probability"] for point in points],
            [point["observed_event_fraction"] for point in points],
            "o-",
            color="#25869b",
            label="Selected model",
        )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed event fraction", title="Test-set reliability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "calibration.png", dpi=140)
    plt.close(figure)


def train_and_evaluate() -> dict:
    if not PROCESSED_PATH.is_file():
        raise FileNotFoundError("Processed daily data is missing. Run scripts\\build_dataset.py first.")
    data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw_targets = pd.to_numeric(data[TARGET], errors="coerce")
    invalid_targets = data[TARGET].notna() & (
        raw_targets.isna() | ~raw_targets.isin([0, 1])
    )
    if invalid_targets.any():
        raise ValueError(f"{TARGET} contains non-binary labels; expected 0/1 or missing.")
    data[TARGET] = raw_targets
    data = data.dropna(subset=["date", TARGET]).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError("At least 50 daily observations are required for a chronological train/validation/test split.")
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    n_train = int(len(data) * 0.6)
    n_validation = int(len(data) * 0.15)
    n_calibration = int(len(data) * 0.15)
    train_slice = slice(0, n_train)
    calibration_slice = slice(n_train, n_train + n_calibration)
    validation_slice = slice(
        n_train + n_calibration, n_train + n_calibration + n_validation
    )
    test_slice = slice(n_train + n_validation + n_calibration, len(data))
    X_train, y_train = X.iloc[train_slice], y.iloc[train_slice]
    X_validation, y_validation = X.iloc[validation_slice], y.iloc[validation_slice]
    X_calibration, y_calibration = X.iloc[calibration_slice], y.iloc[calibration_slice]
    X_test, y_test = X.iloc[test_slice], y.iloc[test_slice]
    if y_train.nunique() < 2:
        raise ValueError("Training partition contains one class; model fitting is not supported.")

    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=SEED, max_iter=1000)),
        ]
    )
    balanced_baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=SEED, max_iter=1000, class_weight="balanced"
                ),
            ),
        ]
    )
    candidate = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5,
                class_weight="balanced", random_state=SEED, n_jobs=1
            )),
        ]
    )
    estimators = {
        "logistic_regression": baseline,
        "logistic_regression_balanced": balanced_baseline,
        "random_forest": candidate,
    }
    training_cv = {}
    splitter = TimeSeriesSplit(n_splits=5)
    for name, estimator in estimators.items():
        fold_scores = []
        for fold, (fit_indices, score_indices) in enumerate(splitter.split(X_train), start=1):
            fold_X, fold_y = X_train.iloc[fit_indices], y_train.iloc[fit_indices]
            if fold_y.nunique() < 2:
                continue
            fold_estimator = clone(estimator)
            fold_estimator.fit(fold_X, fold_y)
            fold_probabilities = fold_estimator.predict_proba(
                X_train.iloc[score_indices]
            )[:, 1]
            fold_scores.append(
                {
                    "fold": fold,
                    "training_samples": int(len(fit_indices)),
                    "validation_samples": int(len(score_indices)),
                    "training_end": data.iloc[fit_indices[-1]]["date"].date().isoformat(),
                    "validation_start": data.iloc[score_indices[0]]["date"].date().isoformat(),
                    "validation_end": data.iloc[score_indices[-1]]["date"].date().isoformat(),
                    "brier_score": float(
                        brier_score_loss(y_train.iloc[score_indices], fold_probabilities)
                    ),
                }
            )
        training_cv[name] = {
            "folds": fold_scores,
            "mean_brier_score": (
                float(np.mean([item["brier_score"] for item in fold_scores]))
                if fold_scores
                else None
            ),
        }
    validation_probabilities = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        validation_probabilities[name] = estimator.predict_proba(X_validation)[:, 1]
    validation_raw_brier: dict[str, float] = {
        name: float(brier_score_loss(y_validation, probabilities))
        for name, probabilities in validation_probabilities.items()
    }
    calibrated_estimators = {}
    calibration_brier_by_model = {}
    calibration_samples = len(y_calibration)
    calibration_sufficient = len(y_calibration) >= 30 and y_calibration.nunique() == 2
    for name, estimator in estimators.items():
        if calibration_sufficient:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(estimator), method="sigmoid"
                )
                calibrated.fit(X_calibration, y_calibration)
            calibrated_estimators[name] = calibrated
            calibration_brier_by_model[name] = float(
                brier_score_loss(
                    y_calibration, calibrated.predict_proba(X_calibration)[:, 1]
                )
            )
        else:
            calibrated_estimators[name] = estimator
            calibration_brier_by_model[name] = None

    validation_calibrated_probabilities = {
        name: estimator.predict_proba(X_validation)[:, 1]
        for name, estimator in calibrated_estimators.items()
    }
    validation_scores = {
        name: float(brier_score_loss(y_validation, probabilities))
        for name, probabilities in validation_calibrated_probabilities.items()
    }
    selected_name = min(validation_scores, key=lambda name: validation_scores[name])
    selected = estimators[selected_name]
    threshold_selections = {
        name: _select_f1_threshold(y_validation.to_numpy(), probabilities)
        for name, probabilities in validation_calibrated_probabilities.items()
    }
    selected_calibrated = calibration_sufficient
    final_model = calibrated_estimators[selected_name]
    calibration_brier = calibration_brier_by_model[selected_name]

    comparison = {}
    for name, estimator in estimators.items():
        comparison[name] = _metrics(y_test, estimator.predict_proba(X_test)[:, 1])
    calibrated_comparison = {
        name: _metrics(
            y_test,
            estimator.predict_proba(X_test)[:, 1],
            threshold_selections[name]["threshold"],
        )
        for name, estimator in calibrated_estimators.items()
    }
    selected_probabilities = final_model.predict_proba(X_test)[:, 1]
    selected_threshold = threshold_selections[selected_name]["threshold"]
    selected_metrics = _metrics(y_test, selected_probabilities, selected_threshold)
    selected_fixed_threshold_metrics = _metrics(
        y_test, selected_probabilities, THRESHOLD
    )
    risk_distribution = {
        "low": int((selected_probabilities < selected_threshold).sum()),
        "moderate": int(
            (
                (selected_probabilities >= selected_threshold)
                & (selected_probabilities < 0.7)
            ).sum()
        ),
        "high": int((selected_probabilities >= 0.7).sum()),
    }
    calibration_points = _calibration_points(y_test, selected_probabilities)
    heldout_brier_skill = (
        float(
            1
            - selected_metrics["brier_score"]
            / brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))
        )
        if y_test.nunique() == 2
        else None
    )
    base_estimator = selected.named_steps["model"]
    if hasattr(base_estimator, "feature_importances_"):
        raw_importance = base_estimator.feature_importances_
    else:
        raw_importance = np.abs(base_estimator.coef_[0])
    total_importance = float(np.sum(raw_importance))
    importance = {
        feature: float(value / total_importance) if total_importance else 0.0
        for feature, value in sorted(
            zip(FEATURES, raw_importance), key=lambda item: item[1], reverse=True
        )
    }
    quality_path = PROCESSED_PATH.parent / "data_quality.json"
    provenance_path = PROCESSED_PATH.parents[1] / "manifests" / "provenance.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    training_dataset = {
        "data_status": provenance.get("data_status", "unverified_import"),
        "source_name": provenance.get("source_name"),
        "filename": provenance.get("filename"),
        "sha256": provenance.get("sha256"),
        "acquisition_date": provenance.get("acquisition_date"),
        "identity_recorded": bool(provenance.get("sha256")),
    }
    target_status = quality.get("target_status", "derived_proxy")
    target_definition = quality.get("target_definition") or TARGET_DEFINITION
    target_source = quality.get("target_source")
    model_version = f"{selected_name}-seed{SEED}-v2"
    bundle = {
        "model": final_model,
        "model_version": model_version,
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "features": FEATURES,
        "reference_values": {key: float(value) for key, value in X_train.median().items()},
        "feature_ranges": {
            key: [float(X_train[key].min()), float(X_train[key].max())] for key in FEATURES
        },
        "training_samples": int(len(X_train)),
        "calibration_samples": int(calibration_samples),
        "calibration_brier": calibration_brier,
        "heldout_test_brier": float(selected_metrics["brier_score"]),
        "heldout_test_samples": int(len(y_test)),
        "heldout_test_has_both_classes": bool(y_test.nunique() == 2),
        "heldout_test_roc_auc": selected_metrics["roc_auc"],
        "heldout_test_brier_skill": heldout_brier_skill,
        "calibration_status": "sigmoid_calibrated_on_calibration_partition" if selected_calibrated else "uncalibrated_insufficient_calibration_evidence",
        "data_status": provenance.get("data_status", "unverified_import"),
        "training_dataset": training_dataset,
        "target_status": target_status,
        "target_definition": target_definition,
        "target_source": target_source,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    _save_plots(y_test, selected_probabilities, importance)
    metrics = {
        "model_version": model_version,
        "selected_model": selected_name,
        "classification_threshold": selected_threshold,
        "data_status": provenance.get("data_status", "unverified_import"),
        "training_dataset": training_dataset,
        "features": FEATURES,
        "synthetic_warning": (
            "Synthetic development data; metrics are not evidence of real-world maritime performance."
            if provenance.get("data_status") == "synthetic"
            else None
        ),
        "target": TARGET,
        "target_status": target_status,
        "target_definition": target_definition,
        "target_source": target_source,
        "prediction_horizon": "Following UTC calendar day",
        "calibration": {
            "status": bundle["calibration_status"],
            "method": "sigmoid",
            "calibration_fit_brier_score": calibration_brier,
            "sample_count": int(calibration_samples),
            "independent_test_brier_score": float(selected_metrics["brier_score"]),
            "fit_brier_by_model": calibration_brier_by_model,
        },
        "model_settings": {
            "logistic_regression": {
                "class_weight": None,
                "selection_metric": "calibrated validation Brier score",
            },
            "logistic_regression_balanced": {
                "class_weight": "balanced",
                "selection_metric": "calibrated validation Brier score",
            },
            "random_forest": {
                "class_weight": "balanced",
                "selection_metric": "calibrated validation Brier score",
            },
        },
        "splits": {
            "train": {"sample_count": int(len(y_train)), "period": {"start": data.iloc[0]["date"].date().isoformat(), "end": data.iloc[n_train - 1]["date"].date().isoformat()}},
            "calibration": {"sample_count": int(len(y_calibration)), "period": {"start": data.iloc[n_train]["date"].date().isoformat(), "end": data.iloc[n_train + n_calibration - 1]["date"].date().isoformat()}},
            "validation": {"sample_count": int(len(y_validation)), "period": {"start": data.iloc[n_train + n_calibration]["date"].date().isoformat(), "end": data.iloc[n_train + n_calibration + n_validation - 1]["date"].date().isoformat()}},
            "test": {"sample_count": int(len(y_test)), "period": {"start": data.iloc[n_train + n_validation + n_calibration]["date"].date().isoformat(), "end": data.iloc[-1]["date"].date().isoformat()}},
        },
        "validation_selection_brier": validation_scores,
        "validation_raw_brier": validation_raw_brier,
        "validation_threshold_metrics": threshold_selections,
        "training_only_temporal_cv": training_cv,
        "comparison_test_raw_models": comparison,
        "comparison_test_calibrated_models": calibrated_comparison,
        "threshold_selection": {
            "method": "Maximum F1 on the chronological validation partition using probabilities calibrated on the separate calibration partition; ties select the lowest threshold.",
            "partition": "validation",
            "sample_count": int(len(y_validation)),
            "positive_count": int(y_validation.sum()),
            "model_thresholds": threshold_selections,
            "selected_model": selected_name,
            "selected_threshold": selected_threshold,
        },
        "test": {
            "sample_count": int(len(y_test)),
            "period": metrics_period(data, test_slice),
            "selected_model": selected_metrics,
            "selected_model_fixed_0_5": selected_fixed_threshold_metrics,
            "selected_model_is_calibrated": selected_calibrated,
            "brier_skill_score": heldout_brier_skill,
            "risk_distribution": risk_distribution,
        },
        "calibration_curve": {
            "points": calibration_points,
            "unavailable_reason": (
                None
                if calibration_points is not None
                else "A calibration curve requires both classes in the final test set."
            ),
        },
        "class_distribution": {
            "train": {"negative": int((y_train == 0).sum()), "positive": int((y_train == 1).sum())},
            "calibration": {"negative": int((y_calibration == 0).sum()), "positive": int((y_calibration == 1).sum())},
            "validation": {"negative": int((y_validation == 0).sum()), "positive": int((y_validation == 1).sum())},
            "test": selected_metrics["class_distribution"],
        },
        "global_feature_importance": importance,
        "feature_importance_warning": "Importance indicates model association, not causation.",
        "data_quality": quality,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def metrics_period(data, split):
    return {
        "start": data.iloc[split.start]["date"].date().isoformat(),
        "end": data.iloc[split.stop - 1]["date"].date().isoformat(),
    }
