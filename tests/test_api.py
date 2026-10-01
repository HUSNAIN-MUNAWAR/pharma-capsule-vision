from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defect_detector.api import create_app


def test_health_and_readiness_without_model() -> None:
    with TestClient(create_app(model_path=None)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is False
        assert client.get("/ready").status_code == 503


def test_predict_requires_model() -> None:
    image = Image.new("RGB", (16, 16), (100, 100, 100))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    with TestClient(create_app(model_path=None)) as client:
        response = client.post(
            "/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 503
