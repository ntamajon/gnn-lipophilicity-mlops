from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import DATA_DIR
from src.dataset import TournamentDataset
from src.predict import load_trained_model, predict_graph


# Creamos la aplicación FastAPI.
app = FastAPI(
    title="GNN Lipophilicity Prediction API",
    description="API for predicting molecular lipophilicity using a Graph Neural Network.",
    version="1.0.0",
)


# Cargamos el dataset una sola vez al arrancar la API.
dataset = TournamentDataset(root=str(DATA_DIR))

# Cargamos el modelo una sola vez al arrancar la API.
model, checkpoint, device = load_trained_model()


class PredictByIndexRequest(BaseModel):
    """
    Esquema de entrada para predecir usando un índice del dataset.
    """

    index: int = Field(
        ...,
        ge=0,
        description="Index of the molecular graph in the processed dataset.",
    )


@app.get("/")
def root():
    """
    Endpoint raíz para comprobar que la API está activa.
    """

    return {
        "message": "GNN Lipophilicity Prediction API",
        "model_name": checkpoint["model_name"],
        "status": "running",
    }


@app.get("/health")
def health():
    """
    Endpoint de salud usado para comprobar que el servicio responde.
    """

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dataset_size": len(dataset),
    }


@app.post("/predict-by-index")
def predict_by_index(request: PredictByIndexRequest):
    """
    Predice la lipofilicidad de una molécula usando un índice del dataset procesado.

    Este endpoint sirve para validar el flujo de inferencia:
    input JSON -> carga del grafo -> predicción del modelo -> respuesta JSON.
    """

    index = request.index

    if index >= len(dataset):
        raise HTTPException(
            status_code=400,
            detail=f"Index {index} is out of range. Dataset size is {len(dataset)}.",
        )

    graph = dataset[index]

    prediction = predict_graph(
        model=model,
        graph=graph,
        device=device,
    )

    return {
        "index": index,
        "prediction": prediction,
        "target": "lipophilicity",
        "model_name": checkpoint["model_name"],
        "best_rmse": checkpoint["best_rmse"],
    }