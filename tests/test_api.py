from fastapi.testclient import TestClient
from app.main import app, metadata

client = TestClient(app)

# Test health
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

# Test metadata
def test_metadata_has_30_features():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert len(data["feature_names"]) == 30

# Test predict với dữ liệu hợp lệ
def test_predict_valid():
    features = {
        name: 1.0
        for name in metadata["feature_names"]
    }
    response = client.post(
        "/predict",
        json={"features": features}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "probability_malignant" in data
    assert "probability_benign" in data

# Test thiếu feature
def test_predict_rejects_missing_features():
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": 10.0
            }
        }
    )
    assert response.status_code == 422