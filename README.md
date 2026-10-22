# Predicting Port Congestion and Shipment Delays Using AI

An AI-powered web application that analyzes port activity data to predict port congestion and shipment delays using machine learning. The system provides historical insights, model performance metrics, and explainable predictions through an interactive dashboard.

## Overview

Port congestion and shipment delays can affect shipping schedules, logistics costs, and supply chain efficiency. This project explores how machine learning can help identify congestion patterns and estimate shipment delay risks using historical port activity data.

The application combines a React frontend with a FastAPI backend to provide interactive data analysis, predictions, and model evaluation.

## Features

- **Interactive Dashboard:** View port activity summaries and congestion statistics.
- **Port Analysis:** Explore historical port activity and congestion trends across selected date ranges.
- **Congestion Prediction:** Generate predictions using the available machine learning model.
- **Shipment Delay Prediction:** Estimate shipment delay risk based on available input features and trained models.
- **Model Performance:** Review evaluation metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and Brier score where applicable.
- **Dataset Management:** Load and analyze supported datasets.
- **Explainable Results:** Present relevant factors and metrics to help interpret model predictions.
- **Date-Range Filtering:** Analyze historical data over selected periods.

## Technology Stack

### Frontend
- React
- TypeScript
- Vite
- JavaScript, HTML, and CSS

### Backend
- Python
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn

### Data and Machine Learning
- Historical port activity and vessel-related features
- Data preprocessing and feature engineering
- Classification models for congestion and delay prediction
- Model evaluation and probability-based predictions

## Project Structure

```text
port-congestion-prediction/
├── backend/
│   ├── app/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── artifacts/
│   ├── models/
│   └── metrics/
├── data/
│   ├── processed/
│   └── synthetic/
├── render.yaml
├── .gitignore
└── README.md
```

## Deployment architecture

This repository is a monorepo with **two separate deployments**:

- Vercel hosts the static React/Vite frontend from `frontend/`.
- Render runs the FastAPI application from `backend/app/main.py`, with model artifacts and processed baseline data stored at the repository root under `artifacts/` and `data/`.

The frontend uses `VITE_API_BASE_URL` to call the HTTPS Render service. It does not host or proxy the Python API. Dataset uploads (up to 25 MiB per CSV) update the active processed data and delay model, so Render needs a persistent disk for those files to survive restarts and deployments. The supplied [render.yaml](./render.yaml) configures a 1 GB persistent disk mounted at `/var/data`; this requires a paid Render web-service plan. Render disks are instance-local and constrain the service to a single instance; disk-backed deploys also do not provide zero-downtime deploys. If those limits are unsuitable, use external object/database storage before enabling dataset uploads at scale.

At local frontend development, the API defaults to `http://127.0.0.1:8000`; no frontend environment file is required. If needed, set a different `VITE_API_BASE_URL` in `frontend/.env.local`. For production, configure the HTTPS Render URL in Vercel; the Vite variable is embedded in browser code and must never contain credentials.

### Deploy the backend to Render

Create a **Web Service** from this repository, or create a Blueprint from `render.yaml`. For manual setup use:

- Root Directory: repository root (`.`), so the model/data directories outside `backend/` remain available.
- Runtime: Python.
- Build Command: `pip install -r backend/requirements.txt`
- Start Command: `uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT`
- Health Check Path: `/ready`
- Python: `.python-version` selects Python 3.14.
- Persistent disk: mount at `/var/data`, at least 1 GB. Set `PORT_DATA_DIR=/var/data/data` and `PORT_ARTIFACTS_DIR=/var/data/artifacts`.

Render provides `PORT` automatically. `render.yaml` declares the Python version, data/artifact paths, disk, and start/build commands. Its `PORT_CORS_ORIGINS` value is intentionally a dashboard prompt (`sync: false`): after the Vercel site exists, set it to the exact frontend origin, for example `https://your-frontend-project.vercel.app`. To retain local frontend access on the deployed API, comma-separate the desired local origins and production origin. Do not add a trailing slash to an origin.

At first startup with an empty mounted disk, the service copies the checked-in processed dataset, provenance, quality report, models, and evaluation JSON into the disk. Subsequent startups preserve the active files. Dataset activation and delay model training can then update files under the disk-backed paths. Startup errors are not swallowed; `/ready` returns 503 if the congestion model or processed data is unavailable.

### Deploy the frontend to Vercel

Import the same GitHub repository as a Vercel project and set:

- Root Directory: `frontend` (leave “Include files outside the Root Directory” off).
- Framework Preset: Vite.
- Build Command: `npm run build`.
- Output Directory: `dist`.
- Install Command: `npm install` (or Vercel's automatic install).
- Environment Variable: `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`.

Select Node.js `22.x` in Vercel (Vite 8 requires Node.js 20.19+ or 22.12+). Set the variable for Production and Preview as appropriate, then redeploy after changes. The frontend includes [`.env.example`](./frontend/.env.example) with a placeholder; replace it with the Render URL and do not treat the example as a live value. The value is public browser configuration, not a secret. API requests fail with a clear configuration error when it is missing, and production rejects non-HTTPS API URLs. The Vercel rewrite serves `index.html` for frontend navigation; API requests go directly from the browser to Render. CORS allows exact origins only: add each Vercel preview origin that needs API access to Render's `PORT_CORS_ORIGINS`; do not use a wildcard.

### API endpoints

The frontend dashboard loads `GET /api/v1/dashboard` and `GET /api/v1/delay/metrics`; the prediction forms submit to `POST /api/v1/predict` and `POST /api/v1/delay/predict`. Dataset management uses `GET /api/v1/datasets/active`, `POST /api/v1/datasets/validate`, and `POST /api/v1/datasets/upload`. Other supported routes are:

```text
GET  /health
GET  /ready
GET  /api/v1/summary
GET  /api/v1/metadata
GET  /api/v1/metrics
GET  /api/v1/ports
GET  /api/v1/ports/{port_id}/history
```

After both services deploy, open `https://<render-service>.onrender.com/ready` and confirm it returns HTTP 200 with model/data loaded. Then use that Render HTTPS URL as `VITE_API_BASE_URL`, redeploy Vercel, and inspect browser developer tools → Network to confirm dashboard requests return JSON from the Render domain with no CORS errors.

## Getting Started

### Prerequisites

Install the following tools before running the application:

- Python 3.14 (the checked-in `.python-version` and Render configuration use 3.14).
- Node.js 20.19+ or 22.12+ and npm (use Node.js 22.x on Vercel).
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/keathansingh/Predicting-Port-Congestion-and-Shipment-Delays-Using-AI.git
cd Predicting-Port-Congestion-and-Shipment-Delays-Using-AI
```

### 2. Set Up the Backend

Create and activate a Python virtual environment from the project root.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Start the FastAPI server:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

The backend will usually be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### 3. Set Up the Frontend

Open a second terminal and navigate to the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal, usually http://localhost:5173.

If your frontend uses a different port, use the URL displayed in the terminal and ensure the backend CORS configuration allows that origin.

## Dataset

The checked-in runtime assets include the processed daily CSV and trained congestion and delay model artifacts. The 13 MB source dataset is ignored by Git and is not needed for inference. Dataset uploads and updated model files are held on the Render persistent disk.

**Data provenance:** Synthetic data is not real-world operational AIS data. If a synthetic dataset is used, its predictions and evaluation results should be interpreted as experimental rather than proof of real-world performance.

For reproducible results, document the dataset source, date coverage, preprocessing steps, target-label definitions, and train/test split used in each evaluation.

## Training and inference

From the repository root, install the backend dependencies as described above. To reproduce the bundled synthetic development dataset and train both classifiers:

```powershell
python scripts\generate_synthetic_data.py
python scripts\validate_data.py --input data\synthetic\ais_development.csv
python scripts\build_dataset.py --input data\synthetic\ais_development.csv
python scripts\train_model.py
python scripts\train_delay_model.py
python scripts\evaluate_model.py
```

The generated AIS-like records and both targets are simulated development data, not real vessel broadcasts, verified congestion events, or shipment delivery outcomes. The congestion and shipment-delay models are independent. The model scripts write their own files under `artifacts\models\` and evaluation JSON under `artifacts\metrics\`; evaluation summaries use held-out data. The delay classifier does not estimate duration, so its duration field is unavailable.

For local inference, start FastAPI and the Vite frontend using the commands above. The API exposes congestion prediction at `POST /api/v1/predict`, shipment-delay classification at `POST /api/v1/delay/predict`, congestion metrics at `GET /api/v1/metrics`, and delay metrics at `GET /api/v1/delay/metrics`. The frontend sends requests to the local API in development and to the configured Render API in production. When the active dataset changes, restart the backend to load it and retrain the shipment-delay model when valid labels are available.

## Model Evaluation

The application includes model evaluation metrics intended to assess predictive performance, such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Brier score, where applicable

Metric availability depends on the target labels, test data, and trained model. Results should be evaluated using held-out data, and probability estimates should be assessed separately from classification confidence.

## Limitations

- Prediction quality depends on dataset quality, feature coverage, and target-label reliability.
- Synthetic data cannot establish real-world predictive accuracy.
- Historical patterns may not reflect current port conditions.
- Model metrics can vary with the selected dataset and evaluation period.
- Operational deployment requires validation against verified real-world port and shipment outcomes.

## Future Improvements

- Integrate verified real-world AIS and port-operation datasets.
- Incorporate weather forecasts and port capacity information.
- Improve probability calibration and explainability.
- Add systematic model comparison and reproducible evaluation reports.
- Deploy the application with appropriate monitoring and data validation.

## Author

**Keathan Singh**

- GitHub: [@keathansingh](https://github.com/keathansingh)
- LinkedIn: [Keathan Singh](https://www.linkedin.com/in/keathan-singh/)

## License

No license has been specified yet. All rights reserved unless a license is added to this repository.
