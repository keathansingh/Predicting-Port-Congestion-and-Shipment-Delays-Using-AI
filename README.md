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
