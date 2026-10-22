# Dashboard frontend

This React, TypeScript, and Vite dashboard is the frontend for the Port Congestion and Shipment Delay Prediction System. It calls the local FastAPI service at `http://127.0.0.1:8000` by default. The backend allows Vite's default ports 5173 and 5174 for local CORS requests.

From this directory in PowerShell:

```powershell
npm install
npm test
npm run dev
npm run build
npm run lint
```

To use a different local API address, set `VITE_API_BASE_URL` in `.env.local`, for example:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The Dataset Manager validates raw AIS CSV files with required columns `MMSI`, `BaseDateTime`, `LAT`, `LON`, and `SOG` before offering processing. Files are limited to 25 MiB and 250,000 rows. Processing produces the active daily port dataset and does not retrain the model. The dashboard reloads one active-data snapshot after upload; its date filters, cards and chart rows are derived from that same snapshot and bounded to its actual coverage. Saved evaluation metrics remain tied to their recorded training dataset.

For the full data pipeline, backend setup, model training, test, and report instructions, see the repository [README](../README.md).
