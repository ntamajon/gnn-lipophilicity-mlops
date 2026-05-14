from fastapi.testclient import TestClient

from src.api_inferencia import app


client = TestClient(app)


def test_root_endpoint():
    """
    Verifica que el endpoint raíz responde correctamente.
    """

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "model_name" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """
    Verifica que el endpoint de salud responde y confirma
    que el modelo y el dataset están disponibles.
    """

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["dataset_size"] > 0


def test_predict_by_index_endpoint():
    """
    Verifica que la API puede hacer una predicción válida
    usando un índice del dataset.
    """

    response = client.post(
        "/predict-by-index",
        json={"index": 0},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["index"] == 0
    assert "prediction" in data
    assert "target" in data
    assert "model_name" in data
    assert "best_rmse" in data

    assert isinstance(data["prediction"], float)
    assert data["target"] == "lipophilicity"


def test_predict_by_index_out_of_range():
    """
    Verifica que la API devuelve error si el índice no existe.
    """

    response = client.post(
        "/predict-by-index",
        json={"index": 999999999},
    )

    assert response.status_code == 400