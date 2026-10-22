from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import CORS_ORIGINS, FEATURES
from app.routes.api import router
from app.schemas import HealthResponse
from app.services.runtime import load_bundle, load_daily
from app.delay_training import load_delay_model
from app.services.persistent_storage import initialize_persistent_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_persistent_storage()
    app.state.model_bundle = load_bundle()
    app.state.delay_model_bundle = load_delay_model()
    daily_data = load_daily()
    app.state.data_available = daily_data is not None and not daily_data.empty
    app.state.data_compatible = bool(
        daily_data is not None and set(FEATURES).issubset(daily_data.columns)
    )
    bundle = app.state.model_bundle
    app.state.model_loaded = bool(
        isinstance(bundle, dict)
        and bundle.get("features") == FEATURES
        and callable(getattr(bundle.get("model"), "predict_proba", None))
    )
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
        "model_loaded": getattr(app.state, "model_loaded", False),
        "delay_model_loaded": bool(delay_bundle and delay_bundle.get("status") == "available"),
        "data_available": getattr(app.state, "data_available", False),
    }


@app.get("/ready", include_in_schema=False)
def readiness():
    model_loaded = bool(getattr(app.state, "model_loaded", False))
    data_available = bool(getattr(app.state, "data_available", False))
    data_compatible = bool(getattr(app.state, "data_compatible", False))
    delay_bundle = getattr(app.state, "delay_model_bundle", None)
    delay_model_loaded = bool(delay_bundle and delay_bundle.get("status") == "available")
    ready = model_loaded and data_available and data_compatible
    result = {
        "status": "ready" if ready else "not_ready",
        "model_loaded": model_loaded,
        "delay_model_loaded": delay_model_loaded,
        "data_available": data_available,
        "data_compatible": data_compatible,
    }
    return JSONResponse(status_code=200 if ready else 503, content=result)
