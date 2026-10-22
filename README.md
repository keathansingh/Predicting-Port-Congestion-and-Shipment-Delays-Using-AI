# Port Congestion and Shipment Delay Prediction System

A local, reproducible prototype for estimating a **next-day AIS-derived port-congestion proxy**. It does not predict verified shipment lateness: no scheduled or actual shipment-delivery timestamps are supplied. Its synthetic development mode is clearly identified throughout the UI and generated files; its model scores are not maritime performance evidence.

## Current data status

The repository still contains only the explicitly synthetic AIS-like development fixture; no raw AIS order was submitted or downloaded. The official [AccessAIS](https://marinecadastre.gov/accessais/) service advertises 2020–2025 data and a 2 GB order limit. A verified estimate for the Los Angeles development bbox and 2023 Q1 returned 43,004 messages (4,379,174 bytes), so a small relevant request is technically manageable. Retrieval requires the portal's order workflow and an email contact, so a data order cannot be completed without a user-provided contact. See `docs\data_provenance.md` for the exact estimate, source terms, and public aggregate product that was not substituted for raw AIS.

## Architecture

```text
Marine Cadastre AIS export (optional) ──┐
                                       ├─> validate/build dataset ─> train/evaluate ─> local model + metrics
Seeded synthetic AIS-like data (demo) ──┘                                      │
                                                                              v
React + TypeScript dashboard <── JSON API ── FastAPI (port history, metrics, predictions)
```

## Requirements

- Python 3.11 or later (the pinned backend requirements were exercised with Python 3.14.7)
- Node.js 20.19+ or 22.12+
- PowerShell on Windows

## Setup and run (PowerShell)

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
python scripts\generate_synthetic_data.py
python scripts\build_dataset.py --input data\synthetic\ais_development.csv
python scripts\train_model.py
python -m uvicorn app.main:app --app-dir backend --reload
```

In a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> (or Vite's fallback <http://localhost:5174> if port 5173 is occupied). The API is at <http://127.0.0.1:8000>; interactive OpenAPI documentation is at <http://127.0.0.1:8000/docs>. Both Vite ports are allowed by the default backend CORS configuration. If the frontend API URL needs changing, set `VITE_API_BASE_URL` in `frontend/.env.local` (for example `http://127.0.0.1:8000`).

### Optional public AIS import

Use an actual CSV export URL copied from the Marine Cadastre portal; an export may be large and have provider-specific availability or access conditions:

```powershell
python scripts\download_data.py --url "PASTE_THE_ACTUAL_EXPORT_URL_HERE" --filename "ais_export.csv" --source-name "Exact portal dataset/export title"
python scripts\build_dataset.py --input data\raw\ais_export.csv --source-name "Exact portal dataset/export title" --source-url "PASTE_THE_ACTUAL_SOURCE_URL_HERE"
```

Raw files are excluded from version control. Review `data\manifests\provenance.json` and the data-quality report after processing. The pipeline currently supports the common Marine Cadastre AIS CSV columns documented in `docs\data_provenance.md`; it fails with a clear error if required columns are absent. URL query strings, fragments, and embedded credentials are stripped from saved provenance to prevent accidentally persisting access tokens.

## Pipeline and model commands

```powershell
python scripts\validate_data.py --input data\synthetic\ais_development.csv
python scripts\build_dataset.py --input data\synthetic\ais_development.csv
python scripts\train_model.py
python scripts\evaluate_model.py
```

The seeded development generator creates a full two-year (2023-01-01 through 2024-12-31 UTC) synthetic AIS-like dataset. It includes varied daily conditions, repeated low-speed candidate-anchorage observations, and explicitly synthetic congestion and delay labels/minutes. Those fields are simulated fixture values, not real AIS or ground-truth shipping outcomes. `build_dataset.py` passes supported delay labels through without using target or duration columns as model inputs. The delay model and congestion model are trained/evaluated as separate artifacts; their test scores only describe synthetic fixture performance.

`build_dataset.py` cleans and spatially filters messages, aggregates vessel and candidate-anchorage activity by UTC day, and calculates dwell only when a candidate vessel has at least two qualifying UTC observations. Days without sufficient dwell evidence remain unavailable instead of being encoded as zero. The congestion proxy is derived from qualifying AIS-like observations; the separately stored `SyntheticCongestionLabel` and `SyntheticDelayLabel` are simulated outcomes. The generator's persistent traffic, weather, queue, and variable within-day spans are fixture assumptions, not real AIS behavior. Raw/synthetic observations, model binaries, and generated chart images are ignored by Git; the small provenance, quality, metrics, and PDF report outputs are retained for reproducibility.

Uploads use the same AIS fields and proxy rule. Congestion preprocessing remains independent of the delay feature: it does not use extra CSV columns such as port capacity, queue length, weather, or delay minutes. The pipeline retains `SyntheticCongestionLabel` separately from the independently calculated next-day AIS-derived proxy. The congestion model target uses the provided synthetic label only when valid labels include both classes; otherwise the AIS proxy remains the model target. The dedicated delay pipeline uses supported current-day port-condition columns and the explicitly provided `SyntheticDelayLabel`, while excluding delay-minute columns. A coordinate match to the configured Los Angeles development geofence identifies that configured scope only; it does not independently verify a real-world port identity.

### Shipment-delay classification

The independent shipment-delay pipeline trains only when `SyntheticDelayLabel` is present, contains valid binary values, and provides both classes across a supported chronological holdout. It aggregates that supplied label by UTC target day (a daily positive means at least one supplied positive label that day) and pairs it with operational values from the preceding UTC day. The classifier uses vessel arrivals, queue length, berth capacity, weather category, wind speed, weekday, and month. `SyntheticDelayLabel`, `SyntheticDelayMinutes`, and `HistoricalDelayMinutes` are excluded from model inputs; no labels are derived from a duration field. Missing predictor values are imputed within the saved sklearn preprocessing pipeline, and class weighting addresses imbalance. The separate `artifacts\models\delay_model.joblib` and `artifacts\metrics\delay_evaluation.json` artifacts do not replace the congestion model or its existing evaluation.

On application startup the delay artifact is checked against the active processed dataset identity and is trained/evaluated when necessary. The held-out test is the latest 20% of labeled daily rows after chronological sorting; earlier 80% is used for training. The final test artifact records the source identity, target distribution, split dates and sample counts, precision, recall, F1, ROC-AUC, confusion matrix, Brier score, and log loss. Delay scores are uncalibrated probabilities (not model confidence). Risk thresholds are Low below 30%, Medium from 30% to below 60%, and High at or above 60%. Delay duration is explicitly unavailable because no independently evaluated duration model is provided. A prior delay evaluation with a different dataset identity is retained as a `delay_evaluation_<identity>.json` historical artifact; congestion evaluation results remain separate. Uploading a dataset does not immediately replace the running model; startup compares the saved artifact with the active dataset identity and trains against newly active valid labels when needed.

The chronological model split is 60% train, 15% calibration fitting, 15% validation/model selection and threshold selection, and 10% final test. Five-fold expanding-window temporal cross-validation is performed inside the training partition only. Unweighted and balanced Logistic Regression and balanced Random Forest are compared by calibrated validation Brier score. Each candidate is sigmoid calibrated on the earlier calibration partition; the classification threshold is then selected by maximum validation F1. The untouched final test reports calibrated validation-threshold comparisons and separately records fixed-0.5 results. Neither calibrator nor threshold selection uses final-test labels.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and artifact availability |
| `GET` | `/api/v1/metadata` | Target, data lineage, model and coverage metadata |
| `GET` | `/api/v1/dashboard` | One consistent active-dataset snapshot for dashboard history, metadata and saved evaluation |
| `GET` | `/api/v1/summary` | Observed dataset and aggregate summary |
| `GET` | `/api/v1/metrics` | Held-out evaluation metrics and chart data |
| `GET` | `/api/v1/delay/metrics` | Separate shipment-delay holdout metrics and model availability |
| `POST` | `/api/v1/delay/predict` | Shipment-delay outcome, uncalibrated probability, risk, factors and recommendations |
| `GET` | `/api/v1/ports` | Available geographic scopes |
| `GET` | `/api/v1/ports/{port_id}/history` | Daily observed/derived indicators |
| `GET` | `/api/v1/datasets/active` | Active source, coverage, row counts and processing metadata |
| `POST` | `/api/v1/datasets/validate` | Validate and preview a raw AIS CSV without activating it |
| `POST` | `/api/v1/datasets/upload` | Validate, process and activate a raw AIS CSV |
| `POST` | `/api/v1/predict` | Next-day congestion probability, separate reliability, explanation and cautions |

The **Shipment Delay** page calls `/api/v1/delay/predict` independently of congestion. Provide daily vessel arrivals, queue, berth capacity, weather, wind speed, and target weekday/month. The endpoint reports an unavailable status if the dedicated model has no valid labeled training data. Its probability is not a confidence score, factors are one-at-a-time model sensitivity rather than causal explanations, and the synthetic target must not be interpreted as real shipment performance.

Summary, dashboard, and port-history GET requests accept optional inclusive `start_date` and `end_date` query parameters in `YYYY-MM-DD` format. Dashboard date presets are anchored to the latest date in active-dataset coverage and request the selected inclusive interval from the API. If the requested period extends beyond the dataset's actual coverage, only available observed rows are returned and the UI reports the shorter coverage; dates are never filled or interpolated.

The React dashboard loads history, active-dataset metadata, the date-filtered summary, and saved evaluation together from `/api/v1/dashboard`. Overview and Port Analysis use that same summary and returned active history. Date-range filters clamp to reported dataset coverage. A successful CSV upload is persisted as the active processed dataset; the dashboard reloads the snapshot without browser caching and confirms its active dataset SHA-256 matches the upload response before replacing displayed state. API responses use `Cache-Control: no-store`. The active analytics dataset, loaded models' training datasets, and saved evaluation datasets are reported separately. Uploads do not immediately replace the running models or rewrite evaluation artifacts; the delay model is rechecked against the active dataset on application startup.

The Dataset Manager accepts a multipart form upload with a required `file` (`.csv`) field and optional `source_name` text field. Raw CSVs must contain `MMSI`, `BaseDateTime`, `LAT`, `LON`, and `SOG`; MMSI values must be nine digits, timestamps parseable, latitude/longitude numeric and within geographic bounds, and speed numeric in the 0–102.3 knot AIS range. Uploads are limited to 25 MiB and 250,000 rows. The validate endpoint returns a first-10-row preview, missing-value/date/numeric/duplicate checks, and warnings. The upload endpoint repeats validation and reports the uploaded coordinate extent. Only datasets whose uploaded coordinates all fall within the configured Los Angeles development geofence receive that configured scope label and its derived next-day anchorage proxy. Other locations are processed as observed daily vessel activity and labeled "geographic area not verified"; anchorage and congestion targets remain unavailable rather than being assigned fabricated zeroes. The congestion pipeline retains its existing next-day target behavior (a valid provided `SyntheticCongestionLabel` when present, otherwise the AIS-derived proxy); neither target is a shipment-delay label. The independent delay pipeline uses `SyntheticDelayLabel` only when supplied and valid. Uploads do not immediately replace running models; a delay model is retrained at startup only if active valid delay labels are available. Invalid input returns HTTP 422, size excess returns 413, and processing failures return an error without claiming activation.

Example request:

```json
{
  "vessel_density": 42,
  "anchorage_vessels": 18,
  "anchorage_dwell_hours": 12,
  "day_of_week": 2,
  "month": 8
}
```

Example PowerShell prediction request (the response is computed from the current trained artifact):

$body = @{
  vessel_density = 42
  anchorage_vessels = 18
  anchorage_dwell_hours = 12
  day_of_week = 2
  month = 8
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/predict" `
  -ContentType "application/json" -Body $body
```

Prediction fails with HTTP 503 if model artifacts have not been trained. It never returns a fabricated fallback prediction.

Prediction requests require all five numeric features, reject unknown fields and out-of-range values, and enforce `anchorage_vessels <= vessel_density`. Values outside the model's observed training ranges are scored with a low-reliability warning rather than presented as in-distribution estimates.

Reliability is a categorical diagnostic, not a correctness probability. Its high/moderate levels require positive held-out Brier skill and ROC-AUC thresholds described in `docs\methodology.md`; failure to meet them is reported as low, while too few samples/classes is insufficient evidence.

## Tests and report

```powershell
python -m pytest backend\tests -q
python scripts\generate_report.py
```

The report generator writes `docs/technical_report.md` and, when Matplotlib's PDF backend is available, `reports/technical_report.pdf`. Both report only metrics found in the actual run artifacts; before training, they explicitly indicate that results are unavailable. Do not interpret synthetic metrics as real-world performance.

## Known limitations

- No actual public AIS download or delivery-delay labels are included.
- The bounding box and candidate-anchorage geofence are coarse, fixed development rules for the Port of Los Angeles area, not verified port operational boundaries.
- Daily AIS signal counts are not equivalent to unique physical vessels, berth assignments, cargo shipments, or port capacity.
- The anchorage-vessel threshold is a proxy chosen to make the event rule explicit and reproducible, not a validated operational threshold.
- AIS gaps, transmission frequency, vessel identity errors, and selection bias can distort activity and dwell estimates.
- A useful deployment requires independently sourced and validated port geofences, arrival/waiting ground truth, more representative historical AIS, and prospective monitoring.

See [methodology](docs/methodology.md), [provenance](docs/data_provenance.md), [limitations](docs/limitations.md), and [demo script](docs/demo_script.md).
