from pathlib import Path

# Ruta raíz del proyecto.
# Path(__file__) apunta a src/config.py.
# .resolve() convierte la ruta a absoluta.
# .parents[1] sube desde src/ hasta la carpeta raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Carpeta donde PyTorch Geometric descargará/procesará el dataset.
DATA_DIR = PROJECT_ROOT / "data" / "tournament"

# Carpeta donde guardaremos modelos entrenados.
MODELS_DIR = PROJECT_ROOT / "models"

# Nombre del modelo candidato seleccionado.
MODEL_NAME = "SAGE_mean"

# Semilla para intentar que los resultados sean reproducibles.
SEED = 42

# Configuración de entrenamiento.
N_SPLITS = 5
MAX_EPOCHS = 300
PATIENCE = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 256

# Ruta final del checkpoint del modelo.
MODEL_PATH = MODELS_DIR / "sage_mean_model.pt"

# Nombre del proyecto en Weights & Biases.
WANDB_PROJECT = "gnn-lipophilicity-mlops"