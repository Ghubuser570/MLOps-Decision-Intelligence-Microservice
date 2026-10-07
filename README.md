# VERDICT: Decision Engine ⚖️

A production-grade, highly available inference microservice for multi-domain risk assessment.

## 🚀 The Architecture
- **API Engine:** FastAPI with strict Pydantic v2 discriminated unions for type-safe inference contracts.
- **Async Execution:** Celery + Redis queues for high-throughput batch processing without blocking the API.
- **ML Models:** LightGBM and CatBoost ensembles, normalized via Isotonic Regression for true probability calibration.
- **Observability:** Prometheus telemetry for endpoint health, and MLflow for model registry and tracking.

## 🛠️ Quickstart

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt