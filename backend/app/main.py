from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.routes.api import router
from app.schemas import HealthResponse
from app.services.runtime import load_bundle, load_daily
from app.delay_training import load_delay_model
from app.services.vercel_dataset_storage import configure_serverless_dataset_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_serverless_dataset_storage()
    app.state.model_bundle = load_bundle()
    app.state.delay_model_bundle = load_delay_model()
    app.state.data_available = load_daily() is not None
    yield


app = FastAPI(
    title="Port Congestion and Shipment Delay Prediction System",
    description="Local API for an explicitly defined AIS-derived congestion proxy.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def prevent_api_caching(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.include_router(router)


@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
@app.get("/health", response_model=HealthResponse)
def health():
    delay_bundle = getattr(app.state, "delay_model_bundle", None)
    return {
        "status": "ok",
        "model_loaded": app.state.model_bundle is not None,
        "delay_model_loaded": bool(delay_bundle and delay_bundle.get("status") == "available"),
        "data_available": app.state.data_available,
    }
