# Methodology

## Prediction target

When no valid supplied congestion labels are available, the modeled event is **the next UTC day having at least 25 distinct MMSIs with AIS observations inside the configured candidate-anchorage geofence**. The Port of Los Angeles development region is approximated by a bounding box; the candidate anchorage rule is a coarse coordinate-and-speed rule. Both are configuration constants, not port authority definitions. The 25-vessel threshold is a transparent prototype threshold, not an empirically calibrated operational congestion standard.

When a valid supplied `SyntheticCongestionLabel` contains both classes, it is retained as the model target; otherwise the model target is the AIS-derived proxy. Both `synthetic_congestion_label` and `ais_congestion_proxy` remain separate daily fields for comparison. The synthetic label is not evidence of verified congestion or delay. The project has no scheduled/actual delivery timestamps or verified shipment outcomes and cannot establish that a shipment was delayed.

## Observation cutoff and horizon

One observation is one UTC calendar day. Features use only observations recorded during that day and calendar values known for the next day. The prediction cutoff is the end of the observed UTC day; the horizon is the following UTC day. The label is shifted from the following day's aggregated anchorage count. The final day without a subsequent observed day is dropped.

## Processing and features

Records are checked for required AIS columns, parseable UTC timestamps, valid geographic ranges, non-null MMSI, duplicate `(MMSI, BaseDateTime)` messages, and configured geographic bounds. Candidate-anchorage messages satisfy the configured geofence and speed `<= 0.5` knots. Daily measures are distinct observed MMSIs, distinct candidate anchorage MMSIs, and median within-day anchorage observation span in hours. A vessel/day must have at least two qualifying observations for a dwell span; a missing daily estimate remains unavailable rather than being changed to zero. Calendar features are next-day weekday and month.

No vessel identifiers, target-derived same-day counts, weather, capacity, delivery data, or future observations enter the features. See the generated quality report for actual row counts and data-quality statistics.

| Feature | Meaning and unit | Source | Missing-value treatment | Available at cutoff |
|---|---|---|---|---|
| `vessel_density` | Distinct MMSIs with valid messages in the current UTC day (vessels/day) | AIS MMSI and BaseDateTime after geofence filter | Required source fields are dropped if missing; resulting daily count is observed | Yes, after current day closes |
| `anchorage_vessels` | Distinct MMSIs satisfying the candidate-anchorage rule during current UTC day (vessels/day) | AIS MMSI, LAT, LON, SOG, and BaseDateTime | Same source-row filtering; no observed candidates yields zero | Yes, after current day closes |
| `anchorage_dwell_hours` | Median within-day first-to-last candidate-anchorage message span (hours), not actual wait | Candidate-anchorage AIS timestamps grouped by MMSI/day | Unavailable if no candidate vessel has at least two qualifying observations | Yes, after current day closes |
| `day_of_week` | Weekday of the predicted next UTC day (Monday=0) | Prediction calendar | Not missing for a valid prediction date | Yes, known in advance |
| `month` | Month of predicted next UTC day (1–12) | Prediction calendar | Not missing for a valid prediction date | Yes, known in advance |

## Model and evaluation

Fixed seed: `42`. Features are median-imputed and standardized in a scikit-learn pipeline for Logistic Regression; Random Forest uses median imputation. Candidates are unweighted Logistic Regression, class-balanced Logistic Regression, and class-balanced Random Forest. Data is split in chronological order into 60% training, 15% calibration-fitting, 15% validation/model-selection and threshold-selection, and 10% final test partitions. Five-fold expanding-window temporal cross-validation is run wholly inside training. Each candidate is calibrated using only the earlier calibration partition; model choice then uses calibrated validation Brier score. The same validation partition selects each calibrated candidate's classification threshold by maximum F1; threshold ties choose the lowest threshold. This deliberate small-sample tradeoff can overfit the validation data, so the final test remains untouched and the report also includes fixed-0.5 results. Neither calibrator nor threshold sees final-test labels. Calibration-fit Brier score is in-sample on the calibration partition and must not be read as an independent calibration estimate.

Report accuracy, precision, recall, F1, ROC-AUC where both classes are represented, Brier score, log loss, class counts, confusion matrix, and a reliability curve when defined. The primary candidate comparison uses calibrated test probabilities and each candidate's validation-selected threshold; raw/fixed-0.5 test comparisons and the selected model's calibrated fixed-0.5 result are also recorded separately. Unsupported metrics are null with an explanation in the JSON. Calibration-fit Brier score is explicitly in-sample; the final test Brier score and calibration curve provide separate held-out probability evidence. The development series is synthetic; its metrics validate software behavior, not maritime performance.

## Probability and reliability

Congestion probability is the calibrated model's estimate of the defined next-day event. Reliability is a separate categorical heuristic (`high`, `moderate`, `low`, `insufficient_evidence`), based on training/calibration/test sample counts, whether Brier skill is positive versus the training-prevalence climatology, held-out ROC-AUC (high ≥0.75; moderate ≥0.60), and whether input features fall outside training ranges. Insufficient samples/classes yield `insufficient_evidence`; out-of-range features or failure to meet both held-out thresholds yield `low`. Synthetic-only training/evaluation always yields `low` reliability even when synthetic holdout metrics exceed heuristic thresholds. The held-out test is not used for model selection, threshold selection, or fitting the model/calibrator. Reliability is not a probability that the model is correct and is not derived as one minus event probability.

## Diagnosis of the original zero-recall result

The prior fixture's final holdout had 10 positives among 74 samples, but the selected calibrated Logistic Regression produced no test probabilities at or above the fixed 0.5 cutoff; its maximum calibrated test probability was 0.198. It therefore returned no positive predictions, yielding zero precision/recall/F1. This was not caused by a missing class-weight setting: the prior Logistic Regression was already class-balanced. The earlier generator independently redrew its latent activity noise each day, weakening the relationship between today's inputs and tomorrow's target; its dwell feature was exactly 12 hours for every row. The earlier partitions also shifted in event prevalence (validation 38.5%, calibration 15.6%, test 13.5%), and the calibration partition came after validation in time. Together these facts show weak predictive separation plus probability/temporal distribution shift, rather than a threshold-only issue. The revised fixture adds explicit persistence and variable simulated spans, and the split now fits calibration before validation. These changes improve the software test fixture only; they do not establish real-world model skill.

## Synthetic fixture design and limitations

The seeded development fixture uses a stationary AR(1) latent activity-noise component (`rho=0.8`), seasonal and weekly activity variation, and explicit synthetic daily arrivals, berth capacity, queue, weather, wind, delay labels, and congestion labels. Repeated UTC observations for the same simulated vessel/day are generated in candidate-anchor coordinates at `SOG <= 0.5` knots, with variable within-day spans; some days have no qualifying anchors. Dwell is calculated only from those qualifying observations and is unavailable for a vessel/day with fewer than two. `SyntheticCongestionLabel` and `SyntheticDelayLabel` are simulated outcomes, not AIS-derived proxy measurements or observed shipping outcomes. The generated records are clearly identified as synthetic development data and are not real vessel broadcasts. These fixture assumptions exercise analytics and model code only; they are not evidence of real maritime behavior or model skill.

## Explanations and action

Global importance is the selected estimator's feature importance or absolute logistic coefficient, normalized for display; it is not causal evidence. Local explanations report one-feature-at-a-time probability differences versus the training median, using the calibrated prediction pipeline. They are model sensitivity indicators; correlated features mean individual changes do not sum to the total prediction. Recommendations are cautious prompts for human review, never automated operational decisions.
