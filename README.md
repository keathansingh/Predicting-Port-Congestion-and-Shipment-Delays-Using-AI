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
│   ├── data/
│   │   ├── processed/
│   │   └── synthetic/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── .gitignore
└── README.md
```

*The structure above is representative. Actual files and folders may vary depending on the current implementation.*

## Deploy frontend and API together on Vercel

The repository is configured for **one Vercel project** rooted at this repository:

- `frontend/dist` is the static Vite build.
- `api/[...path].py` exposes the existing FastAPI routes from the same deployment and domain.
- The checked-in processed daily CSV, provenance, evaluation metrics, and two small sklearn model artifacts are bundled with the function. The 13 MB synthetic source CSV, local uploads, tests, and training-only chart files are excluded from the function bundle because runtime inference uses the processed daily CSV and serialized models.
- `requirements.txt` is the production-only Python dependency manifest. `.python-version` selects Python 3.14, which Vercel currently supports.

From the repository root, import the linked GitHub repository into Vercel as **one project**. Set the Project Name to `Predicting Port Congestion and Shipment Delays Using AI`, Root Directory to `.`, and Framework Preset to `Other`. Leave the build, install, and output settings at their `vercel.json` values. The build installs `frontend/package-lock.json`, runs the frontend production build, and publishes `frontend/dist`; the Python catch-all function routes API paths to FastAPI. `/health` is rewritten to the FastAPI health handler, while `/api/v1/...` paths remain unchanged. Other paths fall back to the SPA entry point.

Connect a **private Vercel Blob store to this same project** (Storage → Blob → Connect Store). Vercel supplies `BLOB_STORE_ID` and its short-lived OIDC token to the function. The Python Blob SDK uses those credentials to store the active uploaded dataset snapshot privately. Do not commit Blob tokens. No `VITE_API_BASE_URL` is needed in production because the frontend calls the API on its own origin; local development keeps the `http://127.0.0.1:8000` default. `PORT_CORS_ORIGINS` is optional and is only needed for intentional cross-origin clients; same-origin Vercel requests do not require CORS.

The dataset upload UI allows up to 4 MiB in production because Vercel Functions limit request and response bodies to 4.5 MB. Local development retains the existing 25 MiB limit. Larger uploads require a client-direct-to-Blob flow; this app has no authentication or authorization flow to securely mint direct-upload tokens, so that capability is not enabled. Dataset activation requires the connected private Blob store; the initial checked-in dataset and predictions remain available without it. Uploaded data is stored across function instances in Blob, but models are not retrained by uploads. When the active dataset no longer matches the bundled delay model, shipment-delay prediction reports that model unavailable until a compatible model is retrained and redeployed.

Deployment checks after each first deployment:

```text
https://<your-vercel-domain>/health
https://<your-vercel-domain>/api/v1/summary
https://<your-vercel-domain>/api/v1/metadata
https://<your-vercel-domain>/api/v1/metrics
https://<your-vercel-domain>/api/v1/delay/metrics
https://<your-vercel-domain>/api/v1/ports
```

The deployment uses Vercel Functions, not a continuously running Uvicorn server. On cold start the function loads the bundled models/data and initializes the private Blob snapshot if one exists. Model training and evaluation remain offline scripts and should be followed by committing/redeploying the updated model and evaluation artifacts.

## Getting Started

### Prerequisites

Install the following tools before running the application:

- Python 3.10 or later
- Node.js and npm
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

The application uses the dataset configured in the project's data pipeline and dataset manager. Depending on the active configuration, the data may include vessel activity, timestamps, port capacity, queue length, weather conditions, and historical delay or congestion labels.

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

For local inference, start FastAPI and the Vite frontend using the commands above. The API exposes congestion prediction at `POST /api/v1/predict`, shipment-delay classification at `POST /api/v1/delay/predict`, congestion metrics at `GET /api/v1/metrics`, and delay metrics at `GET /api/v1/delay/metrics`. The frontend sends requests to the local API in development and to the same deployment origin in production. If a newly activated dataset does not match the bundled delay model, the delay endpoint reports the model unavailable instead of training or writing model files at serverless startup.

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
