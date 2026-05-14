import torch

from src.config import MODEL_PATH, DATA_DIR
from src.dataset import TournamentDataset
from src.models import make_model


def get_device() -> torch.device:
    """
    Selecciona GPU si está disponible.
    Si no hay GPU, usa CPU.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path=MODEL_PATH):
    """
    Carga el modelo entrenado desde un checkpoint.

    El checkpoint contiene:
    - nombre del modelo
    - pesos del modelo
    - configuración de entrenamiento
    - mejor RMSE del fold seleccionado
    """

    device = get_device()

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    dataset = TournamentDataset(root=str(DATA_DIR))

    model_name = checkpoint["model_name"]

    model = make_model(
        model_name=model_name,
        dataset=dataset,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint, device


def predict_graph(model, graph, device):
    """
    Ejecuta predicción para un único grafo de PyTorch Geometric.

    Recibe:
    - modelo cargado
    - grafo molecular en formato Data
    - device CPU/GPU

    Devuelve:
    - predicción numérica de lipofilicidad
    """

    graph = graph.to(device)

    with torch.no_grad():
        prediction = model(graph).view(-1).item()

    return float(prediction)


def predict_dataset_index(index: int):
    """
    Función auxiliar para probar inferencia usando un índice del dataset.

    Esto es útil para tests y para comprobar rápido que el modelo funciona.
    """

    model, checkpoint, device = load_trained_model()

    dataset = TournamentDataset(root=str(DATA_DIR))

    graph = dataset[index]

    prediction = predict_graph(
        model=model,
        graph=graph,
        device=device,
    )

    return {
        "index": index,
        "prediction": prediction,
        "model_name": checkpoint["model_name"],
        "best_rmse": checkpoint["best_rmse"],
    }


if __name__ == "__main__":
    result = predict_dataset_index(0)
    print(result)