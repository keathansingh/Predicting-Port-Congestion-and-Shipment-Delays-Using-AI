# Technical report: Port Congestion and Shipment Delay Prediction System

## 1. Problem definition

This prototype estimates the probability of a next-day AIS-derived congestion proxy for a configured Port of Los Angeles development geofence. It does not estimate verified shipment lateness because scheduled and actual delivery timestamps are unavailable.

## 2. Dataset and provenance

Data status: **synthetic**. Source name: Seeded synthetic AIS-like development data. Public AIS observations acquired: no. Synthetic raw records: 69704; processed daily rows: 729; invalid timestamps: 0; invalid coordinates: 0; duplicates removed: 0. Coverage: 2023-01-02 to 2024-12-30. Synthetic development observations are generated from a fixed seed and are not real AIS.

## 3. Target and assumptions

Positive target: the following UTC day has at least 25 distinct AIS-observed MMSIs in a candidate anchorage geofence, defined by coordinates and speed at or below 0.5 knots. This is a transparent development proxy, not a validated port congestion or shipment-delay label. Prediction uses the preceding UTC day's observations.

## 4. Feature engineering

Features are same-day unique observed-vessel count, same-day unique candidate anchorage-vessel count, median within-day anchorage observation span, and next-day weekday/month. The pipeline removes invalid timestamps/coordinates, duplicate MMSI/timestamp pairs, and records outside the development bounding box. Missing dwell is zero-coded and is not evidence of zero actual waiting.

## 5. Models and evaluation

The fixed-seed unweighted and balanced Logistic Regression baselines and balanced Random Forest candidate use chronological 60%/15%/15%/10% training, calibration, validation, and test splits. Five-fold expanding-window temporal cross-validation occurs only within training (logistic_regression mean expanding-window Brier=0.148, logistic_regression_balanced mean expanding-window Brier=0.196, random_forest mean expanding-window Brier=0.138). Each model is sigmoid calibrated using the earlier calibration partition; model selection uses calibrated validation Brier score. Candidate decision thresholds are then chosen by maximum F1 on that same validation partition. Threshold ties choose the lowest cutoff. No final-test labels are used to choose the model, fit the calibrator, or select a threshold. The report separately retains raw fixed-0.5 comparisons. Calibration-fit Brier score is in-sample on the calibration partition, not an independent calibration estimate.

## 6. Actual results

Evaluation used 74 final chronological test samples (2024-10-18 through 2024-12-30). Selected model: logistic_regression; validation-selected threshold: 0.393. Test metrics at that threshold: sample_count=74, classification_threshold=0.3930822254711993, class_distribution={'negative': 63, 'positive': 11}, accuracy=0.8783783783783784, precision=0.625, recall=0.45454545454545453, f1=0.5263157894736842, roc_auc=0.8672438672438672, brier_score=0.09696863964149284, log_loss=0.3053793087827633, confusion_matrix=[[60, 3], [6, 5]]

Chronological class balance:

| Partition | Samples | Negative | Positive | Positive rate |
|---|---:|---:|---:|---:|
| train | 437 | 320 | 117 | 26.8% |
| calibration | 109 | 78 | 31 | 28.4% |
| validation | 109 | 99 | 10 | 9.2% |
| test | 74 | 63 | 11 | 14.9% |

Before changes, the previous synthetic fixture's uncalibrated Logistic Regression at fixed 0.5 on 74 chronological test samples (10 positive, 64 negative) had precision=0.000, recall=0.000, F1=0.000, ROC-AUC=0.527, Brier=0.138, confusion matrix=`[[64, 0], [10, 0]]`.

The prior generator redrew latent activity noise independently each day and its dwell feature was constant at 12 hours. Logistic Regression was already class-balanced, so missing class weighting was not the cause. Its calibrated test probabilities maxed at 0.198, below the deployed 0.5 cutoff, so it predicted no positive events. Prior label prevalence also shifted from 38.5% in validation to 15.6% in calibration and 13.5% in test, while calibration followed validation chronologically. This is evidence of weak separation and temporal probability shift, not proof that threshold lowering alone would fix it. In a descriptive whole-fixture check, vessel density averaged 52.41 for positive versus 46.43 for negative labels (correlation 0.226); anchorage vessels averaged 21.46 versus 17.59 (correlation 0.187); dwell had zero variance. These are retrospective fixture diagnostics, not causal effects or test-informed feature selection.

The current synthetic fixture was revised to add persistent activity and variable simulated within-day anchorage spans. It is a different synthetic realization and this is not a paired experiment; any apparent score change cannot be attributed to model improvements or generalized to real ports.

Current synthetic positive rates are 28.4% in calibration, 9.2% in validation, and 14.9% in test. This sizable chronological swing illustrates why the threshold and scores are fixture-specific.

The final test set contains 11 positive and 63 negative examples:

### Fixed 0.5 threshold; uncalibrated model outputs

| Model | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.500 | 0.838 | 0.333 | 0.091 | 0.143 | 0.867 | 0.098 |
| logistic_regression_balanced | 0.500 | 0.851 | 0.500 | 0.545 | 0.522 | 0.864 | 0.102 |
| random_forest | 0.500 | 0.851 | 0.500 | 0.636 | 0.560 | 0.859 | 0.108 |

### Validation-selected threshold; calibrated model outputs

| Model | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.393 | 0.878 | 0.625 | 0.455 | 0.526 | 0.867 | 0.097 |
| logistic_regression_balanced | 0.466 | 0.838 | 0.333 | 0.091 | 0.143 | 0.864 | 0.097 |
| random_forest | 0.419 | 0.865 | 0.571 | 0.364 | 0.444 | 0.859 | 0.098 |

The selected model's calibrated-probability Brier score on the final test is 0.09696863964149284; held-out Brier skill versus training-prevalence climatology is 0.3109777335691587. Calibration-fit Brier score on calibration data is 0.16313400255457822 and is in-sample, not an independent estimate.

All computed metrics describe this dataset only. Synthetic-data scores are pipeline demonstration results, not real-world maritime model performance. Unsupported metrics are stored as null with an explanation in the metrics artifact.

## 7. Explainability and operational relevance

Global model importance ranks predictive association and does not establish causality. Local explanations compare calibrated model output with one input feature replaced by its training median; these sensitivities are not additive explanations. Recommendations are advisory and require human review.

## 8. Limitations and future work

No raw AIS file was downloaded or processed. AccessAIS reported an estimated 43,004 records (4,379,174 bytes) for the 2023 Q1 project-area request, below its advertised 2 GB limit, but order retrieval requires a contact email that was not provided. The synthetic fixture is retained as development-only data. Operational port boundaries, berth assignments, port capacity, weather observations, and verified delivery labels are also absent. Validate geography and event definitions with port experts, complete the AccessAIS order and preserve its exact file/terms, assess calibration and drift prospectively, and obtain actual shipment outcomes before investigating shipment-delay prediction.
