from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from verdict.schemas import BatchInferenceRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verdict.api")

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 VERDICT API is starting up...")
    yield
    logger.info("🛑 VERDICT API is shutting down...")

app = FastAPI(
    title="VERDICT: Decision Engine",
    description="Probabilistic inference service for high-stakes risk assessment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.get("/")
async def root():
    return {
        "service": "VERDICT API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }

@app.post("/api/v1/predict", response_model=list[PredictionResponse])
async def predict_batch(request: BatchInferenceRequest):
    """
    Unified multi-domain inference endpoint. 
    Returning mock data for Skeleton Phase verification.
    """
    return [
        {
            "decision": "ACCEPT",
            "probability": 0.85,
            "uncertainty": 0.02,
            "cost_risk": 150.50
        }
    ]