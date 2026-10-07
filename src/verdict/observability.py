import os
import logging
import mlflow
from prometheus_client import Counter, Histogram

logger = logging.getLogger("verdict.observability")

# Prometheus Metrics
INFERENCE_REQUEST_COUNT = Counter(
    "verdict_inference_requests_total",
    "Total number of inference requests received",
    ["dataset", "status"]
)

INFERENCE_LATENCY = Histogram(
    "verdict_inference_latency_seconds",
    "Latency of inference requests in seconds",
    ["dataset"]
)

DECISION_COUNT = Counter(
    "verdict_decisions_total",
    "Total number of decisions made by the engine",
    ["dataset", "decision"]
)

def setup_mlflow_tracking():
    """Initializes MLflow experiment tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("VERDICT_Production_Engine")
    logger.info(f"MLflow tracking initialized at {tracking_uri}")

def log_model_metrics(metrics: dict, step: int = 0):
    """Logs real-time calibration and accuracy metrics to MLflow."""
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}")