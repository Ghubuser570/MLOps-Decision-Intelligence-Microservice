import os
import logging
import joblib
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from celery import Celery

logger = logging.getLogger("verdict.tasks")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("verdict_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# The Lazy Loader Dictionary
_models = {}

def load_engines():
    """Mounts the models safely inside the child process to prevent Segmentation Faults."""
    if not _models:
        _models['lgb'] = lgb.Booster(model_file="models/lightgbm_model.bin")
        _models['cat'] = CatBoostClassifier().load_model("models/catboost_model.bin")
        _models['cal'] = joblib.load("models/calibrator.joblib")
        _models['prep'] = joblib.load("models/preprocessor.joblib")
        logger.info("🧠 VERDICT AI Engines loaded safely post-fork.")
    return _models['lgb'], _models['cat'], _models['cal'], _models['prep']

@celery_app.task(bind=True, name="verdict.tasks.process_inference_batch")
def process_inference_batch(self, payload: dict) -> list[dict]:
    instances = payload.get("instances", [])
    if not instances:
        return []

    try:
        lgb_m, cat_m, cal_m, prep_m = load_engines()
    except Exception as e:
        logger.error(f"⚠️ Engine Load Error: {e}")
        return [{"decision": "ABSTAIN", "probability": 0.0, "uncertainty": 1.0, "cost_risk": 0.0} for _ in instances]

    df = pd.DataFrame(instances)
    if "dataset_type" in df.columns:
        df = df.drop(columns=["dataset_type"])

    # Math processing
    X_processed = prep_m.transform(df)
    lgb_preds = lgb_m.predict(X_processed)
    cat_preds = cat_m.predict_proba(X_processed)[:, 1]

    ensemble_preds = (lgb_preds + cat_preds) / 2.0
    calibrated_probs = cal_m.predict(ensemble_preds)

    responses = []
    for prob in calibrated_probs:
        prob_val = float(prob)
        decision = "ACCEPT" if prob_val >= 0.50 else "REJECT"
        uncertainty = float(1.0 - abs(prob_val - 0.5) * 2)
        responses.append({
            "decision": decision,
            "probability": round(prob_val, 4),
            "uncertainty": round(uncertainty, 4),
            "cost_risk": round((1.0 - prob_val) * 1000, 2)
        })
        
    return responses