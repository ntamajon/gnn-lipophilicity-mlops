from src.config import DATA_DIR, MODEL_NAME
from src.dataset import TournamentDataset
from src.models import make_model


def test_model_can_be_created():
    """
    Verifica que el modelo candidato puede instanciarse.

    Este test no entrena el modelo.
    Solo comprueba que la arquitectura se puede construir con el dataset.
    """

    dataset = TournamentDataset(root=str(DATA_DIR))

    model = make_model(
        model_name=MODEL_NAME,
        dataset=dataset,
    )

    assert model is not None