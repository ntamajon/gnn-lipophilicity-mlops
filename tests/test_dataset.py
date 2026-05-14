from src.config import DATA_DIR
from src.dataset import TournamentDataset


def test_dataset_loads_correctly():
    """
    Verifica que el dataset se puede cargar correctamente.

    Este test comprueba que:
    - El dataset tiene al menos un grafo.
    - Cada grafo tiene atributos básicos esperados por PyTorch Geometric.
    """

    dataset = TournamentDataset(root=str(DATA_DIR))

    assert len(dataset) > 0

    graph = dataset[0]

    assert hasattr(graph, "x")
    assert hasattr(graph, "edge_index")
    assert hasattr(graph, "y")

    assert graph.x.shape[0] > 0
    assert graph.edge_index.shape[0] == 2