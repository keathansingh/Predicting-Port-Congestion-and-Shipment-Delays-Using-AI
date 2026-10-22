import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import METRICS_PATH, PROVENANCE_PATH, QUALITY_PATH  # noqa: E402

BASELINE_METRICS_PATH = ROOT / "artifacts" / "metrics" / "evaluation_pre_improvement.json"


def format_model_comparison(metrics: dict | None) -> str:
    if not metrics:
        return "Baseline/candidate comparison is unavailable until model training runs."
    sections = []
    for title, comparisons, threshold_label in (
        ("Fixed 0.5 threshold; uncalibrated model outputs", metrics.get("comparison_test_raw_models"), "0.500"),
        (
            "Validation-selected threshold; calibrated model outputs",
            metrics.get("comparison_test_calibrated_models"),
            "validation-selected",
        ),
    ):
        if not comparisons:
            continue
        rows = [
            f"### {title}",
            "",
            "| Model | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, values in comparisons.items():
            auc = f"{values['roc_auc']:.3f}" if values["roc_auc"] is not None else "N/A"
            cutoff = (
                f"{values['classification_threshold']:.3f}"
                if threshold_label == "validation-selected"
                else threshold_label
            )
            rows.append(
                f"| {name} | {cutoff} | {values['accuracy']:.3f} | "
                f"{values['precision']:.3f} | {values['recall']:.3f} | "
                f"{values['f1']:.3f} | {auc} | {values['brier_score']:.3f} |"
            )
        sections.append("\n".join(rows))
    return "\n\n".join(sections) or "Model comparison is unavailable."


def format_class_distribution(metrics: dict | None) -> str:
    if not metrics:
        return "Chronological partition class counts are unavailable."
    rows = [
        "| Partition | Samples | Negative | Positive | Positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "calibration", "validation", "test"):
        counts = metrics["class_distribution"][name]
        total = counts["negative"] + counts["positive"]
        rate = counts["positive"] / total if total else 0
        rows.append(
            f"| {name} | {total} | {counts['negative']} | "
            f"{counts['positive']} | {rate:.1%} |"
        )
    return "\n".join(rows)


def main():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")) if PROVENANCE_PATH.exists() else {}
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8")) if QUALITY_PATH.exists() else {}
    baseline = (
        json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
        if BASELINE_METRICS_PATH.exists()
        else None
    )
    score_summary = (
        f"Evaluation used {metrics['test']['sample_count']} final chronological test samples "
        f"({metrics['test']['period']['start']} through {metrics['test']['period']['end']}). "
        f"Selected model: {metrics['selected_model']}; validation-selected threshold: "
        f"{metrics['threshold_selection']['selected_threshold']:.3f}. Test metrics at that threshold: "
        + ", ".join(f"{name}={value}" for name, value in metrics["test"]["selected_model"].items() if value is not None)
        if metrics
        else "No model evaluation artifact exists. Run scripts\\train_model.py; no performance scores are reported."
    )
    model_comparison = format_model_comparison(metrics)
    class_distribution = format_class_distribution(metrics)
    cv_summary = (
        ", ".join(
            f"{name} mean expanding-window Brier={scores['mean_brier_score']:.3f}"
            for name, scores in metrics["training_only_temporal_cv"].items()
            if scores["mean_brier_score"] is not None
        )
        if metrics
        else "Training-only temporal cross-validation is unavailable."
    )
    baseline_summary = "No preserved pre-improvement evaluation artifact was available."
    baseline_diagnosis = ""
    if baseline:
        baseline_lr = baseline["comparison_test_raw_models"]["logistic_regression"]
        baseline_summary = (
            f"Before changes, the previous synthetic fixture's uncalibrated Logistic Regression "
            f"at fixed 0.5 on {baseline_lr['sample_count']} chronological test samples "
            f"({baseline_lr['class_distribution']['positive']} positive, "
            f"{baseline_lr['class_distribution']['negative']} negative) had precision="
            f"{baseline_lr['precision']:.3f}, recall={baseline_lr['recall']:.3f}, "
            f"F1={baseline_lr['f1']:.3f}, ROC-AUC={baseline_lr['roc_auc']:.3f}, "
            f"Brier={baseline_lr['brier_score']:.3f}, confusion matrix="
            f"`{baseline_lr['confusion_matrix']}`."
        )
        baseline_diagnosis = (
            "The prior generator redrew latent activity noise independently each day and "
            "its dwell feature was constant at 12 hours. Logistic Regression was already "
            "class-balanced, so missing class weighting was not the cause. Its calibrated "
            "test probabilities maxed at 0.198, below the deployed 0.5 cutoff, so it "
            "predicted no positive events. Prior label prevalence also shifted from 38.5% "
            "in validation to 15.6% in calibration and 13.5% in test, while calibration "
            "followed validation chronologically. This is evidence of weak separation and "
            "temporal probability shift, not proof that threshold lowering alone would fix it."
            " In a descriptive whole-fixture check, vessel density averaged 52.41 for "
            "positive versus 46.43 for negative labels (correlation 0.226); anchorage "
            "vessels averaged 21.46 versus 17.59 (correlation 0.187); dwell had zero "
            "variance. These are retrospective fixture diagnostics, not causal effects or "
            "test-informed feature selection."
        )
    current_prevalence = metrics["class_distribution"] if metrics else None
    prevalence_shift = ""
    if current_prevalence:
        rates = {
            name: counts["positive"] / sum(counts.values())
            for name, counts in current_prevalence.items()
        }
        prevalence_shift = (
            f"Current synthetic positive rates are {rates['calibration']:.1%} in "
            f"calibration, {rates['validation']:.1%} in validation, and "
            f"{rates['test']:.1%} in test. This sizable chronological swing illustrates "
            "why the threshold and scores are fixture-specific."
        )
    text = f"""# Technical report: Port Congestion and Shipment Delay Prediction System

## 1. Problem definition

This prototype estimates the probability of a next-day AIS-derived congestion proxy for a configured Port of Los Angeles development geofence. It does not estimate verified shipment lateness because scheduled and actual delivery timestamps are unavailable.

## 2. Dataset and provenance

Data status: **{provenance.get('data_status', 'not available')}**. Source name: {provenance.get('source_name') or 'none recorded'}. Public AIS observations acquired: {'no' if provenance.get('data_status') == 'synthetic' else 'not established'}. Synthetic raw records: {quality.get('input_rows', 'unavailable')}; processed daily rows: {quality.get('processed_rows', 'unavailable')}; invalid timestamps: {quality.get('invalid_timestamps', 'unavailable')}; invalid coordinates: {quality.get('invalid_coordinates', 'unavailable')}; duplicates removed: {quality.get('duplicate_rows_removed', 'unavailable')}. Coverage: {quality.get('observation_start', 'unavailable')} to {quality.get('observation_end', 'unavailable')}. Synthetic development observations are generated from a fixed seed and are not real AIS.

## 3. Target and assumptions

Positive target: the following UTC day has at least 25 distinct AIS-observed MMSIs in a candidate anchorage geofence, defined by coordinates and speed at or below 0.5 knots. This is a transparent development proxy, not a validated port congestion or shipment-delay label. Prediction uses the preceding UTC day's observations.

## 4. Feature engineering

Features are same-day unique observed-vessel count, same-day unique candidate anchorage-vessel count, median within-day anchorage observation span, and next-day weekday/month. The pipeline removes invalid timestamps/coordinates, duplicate MMSI/timestamp pairs, and records outside the development bounding box. Missing dwell is zero-coded and is not evidence of zero actual waiting.

## 5. Models and evaluation

The fixed-seed unweighted and balanced Logistic Regression baselines and balanced Random Forest candidate use chronological 60%/15%/15%/10% training, calibration, validation, and test splits. Five-fold expanding-window temporal cross-validation occurs only within training ({cv_summary}). Each model is sigmoid calibrated using the earlier calibration partition; model selection uses calibrated validation Brier score. Candidate decision thresholds are then chosen by maximum F1 on that same validation partition. Threshold ties choose the lowest cutoff. No final-test labels are used to choose the model, fit the calibrator, or select a threshold. The report separately retains raw fixed-0.5 comparisons. Calibration-fit Brier score is in-sample on the calibration partition, not an independent calibration estimate.

## 6. Actual results

{score_summary}

Chronological class balance:

{class_distribution}

{baseline_summary}

{baseline_diagnosis}

The current synthetic fixture was revised to add persistent activity and variable simulated within-day anchorage spans. It is a different synthetic realization and this is not a paired experiment; any apparent score change cannot be attributed to model improvements or generalized to real ports.

{prevalence_shift}

The final test set contains {metrics["test"]["selected_model"]["class_distribution"]["positive"] if metrics else "N/A"} positive and {metrics["test"]["selected_model"]["class_distribution"]["negative"] if metrics else "N/A"} negative examples:

{model_comparison}

The selected model's calibrated-probability Brier score on the final test is {metrics["test"]["selected_model"]["brier_score"] if metrics else "unavailable"}; held-out Brier skill versus training-prevalence climatology is {metrics["test"]["brier_skill_score"] if metrics else "unavailable"}. Calibration-fit Brier score on calibration data is {metrics["calibration"]["calibration_fit_brier_score"] if metrics else "unavailable"} and is in-sample, not an independent estimate.

All computed metrics describe this dataset only. Synthetic-data scores are pipeline demonstration results, not real-world maritime model performance. Unsupported metrics are stored as null with an explanation in the metrics artifact.

## 7. Explainability and operational relevance

Global model importance ranks predictive association and does not establish causality. Local explanations compare calibrated model output with one input feature replaced by its training median; these sensitivities are not additive explanations. Recommendations are advisory and require human review.

## 8. Limitations and future work

No raw AIS file was downloaded or processed. AccessAIS reported an estimated 43,004 records (4,379,174 bytes) for the 2023 Q1 project-area request, below its advertised 2 GB limit, but order retrieval requires a contact email that was not provided. The synthetic fixture is retained as development-only data. Operational port boundaries, berth assignments, port capacity, weather observations, and verified delivery labels are also absent. Validate geography and event definitions with port experts, complete the AccessAIS order and preserve its exact file/terms, assess calibration and drift prospectively, and obtain actual shipment outcomes before investigating shipment-delay prediction.
"""
    report_path = ROOT / "docs" / "technical_report.md"
    report_path.write_text(text, encoding="utf-8")
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pdf_path = ROOT / "reports" / "technical_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for page_text in (text[:2600], text[2600:5200], text[5200:]):
                if not page_text.strip():
                    continue
                figure = plt.figure(figsize=(8.27, 11.69))
                figure.text(0.08, 0.95, page_text, va="top", ha="left", wrap=True, fontsize=9)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        print(f"Generated report: {report_path} and {pdf_path}")
    except ImportError:
        print(f"Generated Markdown report: {report_path}; PDF backend unavailable.")


if __name__ == "__main__":
    main()
