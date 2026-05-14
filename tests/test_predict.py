from pathlib import Path

from src.config import MODEL_PATH
from src.predict import predict_dataset_index


def test_model_file_exists():
    """
    Verifica que existe un modelo entrenado guardado.

    Este test asume que ya se ha ejecutado previamente:
    python -m src.train
    """

    assert Path(MODEL_PATH).exists()


def test_prediction_returns_valid_output():
    """
    Verifica que el modelo cargado puede predecir sobre un grafo del dataset.

    No comprobamos el valor exacto de la predicción porque puede variar
    según el entrenamiento, pero sí comprobamos la estructura de salida.
    """

    result = predict_dataset_index(0)

    assert "index" in result
    assert "prediction" in result
    assert "model_name" in result
    assert "best_rmse" in result

    assert result["index"] == 0
    assert isinstance(result["prediction"], float)
    assert isinstance(result["model_name"], str)
    assert isinstance(result["best_rmse"], float)