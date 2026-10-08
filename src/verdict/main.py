import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from verdict.schemas import BatchInferenceRequest, PredictionResponse
from verdict.tasks import process_inference_batch
from verdict.web.app import web_router

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

# Middleware & Exception Handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# Web Dashboard Router
app.include_router(web_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check probe for container orchestrators."""
    return {"status": "healthy", "service": "verdict-api"}


@app.post("/api/v1/predict", response_model=list[PredictionResponse], tags=["Inference"])
@limiter.limit("60/minute")
async def predict(request: Request, body: BatchInferenceRequest):
    """Dispatches inference payload to Celery queue and awaits calibrated decision."""
    try:
        payload_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        task = process_inference_batch.delay(payload_data)
        results = task.get(timeout=10)
        return results
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Engine processing error.")