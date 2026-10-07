import pytest
from fastapi.testclient import TestClient
from verdict.main import app

client = TestClient(app)

def test_root_health():
    """Verify the API is online and responding."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "VERDICT API"

def test_predict_batch_validation():
    """Verify strict Pydantic schema validation blocks bad data."""
    bad_payload = {
        "dataset": "adult",
        "instances": [{"age": "invalid_string", "workclass": "Private"}]
    }
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422  # Unprocessable Entity