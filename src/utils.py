import random
import math

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def seed_everything(seed: int = 42) -> None:
    """
    Fija semillas para reducir la aleatoriedad del experimento.
    Esto mejora la reproducibilidad, aunque en GPU puede seguir habiendo
    pequeñas variaciones por operaciones no deterministas.
    """

    # Semilla del módulo random de Python.
    random.seed(seed)

    # Semilla de NumPy.
    np.random.seed(seed)

    # Semilla de PyTorch en CPU.
    torch.manual_seed(seed)

    # Semilla de PyTorch en todas las GPUs disponibles.
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    """
    Calcula Root Mean Squared Error.
    Es la métrica principal del proyecto.
    """

    return math.sqrt(mean_squared_error(y_true, y_pred))