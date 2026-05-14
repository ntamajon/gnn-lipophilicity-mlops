import json
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    MODEL_NAME,
    MODEL_PATH,
    SEED,
    N_SPLITS,
    MAX_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    WANDB_PROJECT,
)
from src.dataset import TournamentDataset
from src.models import make_model
from src.utils import seed_everything, rmse


def get_device() -> torch.device:
    """
    Selecciona GPU si está disponible.
    Si no, usa CPU.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(
    dataset,
    train_ids,
    val_ids,
    model_name,
    device,
    max_epochs,
    patience,
    learning_rate,
    weight_decay,
    batch_size,
):
    """
    Entrena un modelo para un único fold.

    Recibe:
    - dataset completo
    - índices de entrenamiento
    - índices de validación
    - nombre del modelo
    - configuración de entrenamiento

    Devuelve:
    - modelo con mejores pesos del fold
    - mejor RMSE de validación
    - historial de entrenamiento
    """

    # DataLoader de entrenamiento.
    # shuffle=True porque queremos mezclar grafos en cada época.
    train_loader = DataLoader(
        dataset[train_ids],
        batch_size=batch_size,
        shuffle=True,
    )

    # DataLoader de validación.
    # shuffle=False porque no hace falta mezclar en evaluación.
    val_loader = DataLoader(
        dataset[val_ids],
        batch_size=batch_size,
        shuffle=False,
    )

    # Creamos el modelo usando la factory definida en src/models.py.
    model = make_model(model_name, dataset).to(device)

    # AdamW es una variante de Adam con weight decay desacoplado.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # SmoothL1Loss es Huber Loss.
    # Es más robusta a outliers que MSE.
    criterion = torch.nn.SmoothL1Loss()

    # Reduce el learning rate si el RMSE de validación se estanca.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    # Variables para early stopping.
    best_rmse = float("inf")
    best_state = None
    wait = 0

    # Guardaremos tuplas: epoch, train_loss, val_rmse.
    history = []

    for epoch in range(1, max_epochs + 1):
        # ======================
        # Entrenamiento
        # ======================

        model.train()

        total_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            # Movemos el batch a CPU o GPU.
            batch = batch.to(device)

            # Limpiamos gradientes anteriores.
            optimizer.zero_grad()

            # Target real.
            y_true = batch.y.view(-1).float()

            # Predicción del modelo.
            y_pred = model(batch).view(-1)

            # Cálculo de pérdida.
            loss = criterion(y_pred, y_true)

            # Backpropagation.
            loss.backward()

            # Clipping para evitar gradientes explosivos.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # Actualización de pesos.
            optimizer.step()

            # Acumulamos loss ponderada por número de grafos.
            num_graphs = batch.num_graphs
            total_loss += loss.item() * num_graphs
            total_graphs += num_graphs

        train_loss = total_loss / max(total_graphs, 1)

        # ======================
        # Validación
        # ======================

        model.eval()

        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                y_true = batch.y.view(-1).float().cpu().numpy()
                y_pred = model(batch).view(-1).cpu().numpy()

                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        val_rmse = rmse(y_true_all, y_pred_all)

        # Actualizamos scheduler usando la métrica de validación.
        scheduler.step(val_rmse)

        # Guardamos historial.
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
            }
        )

        # Early stopping: guardamos los mejores pesos observados.
        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        # Logging por consola.
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"{model_name} | epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_rmse:.4f} | "
                f"best_rmse={best_rmse:.4f} | "
                f"wait={wait}"
            )

        # Logging en W&B si está activo.
        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "best_val_rmse": best_rmse,
                }
            )

        # Si no mejora durante demasiadas épocas, paramos.
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restauramos los mejores pesos del fold.
    model.load_state_dict(best_state)

    return model, best_rmse, history


def run_kfold_training(dataset, model_name, device):
    """
    Ejecuta K-Fold Cross-Validation para el modelo seleccionado.

    Devuelve:
    - lista de modelos entrenados
    - lista de RMSE por fold
    - historiales por fold
    """

    train_idx = np.array(dataset.train_idx)

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_models = []
    fold_scores = []
    fold_histories = []

    for fold, (train_pos, val_pos) in enumerate(kfold.split(train_idx), start=1):
        print(f"\n========== {model_name} | Fold {fold}/{N_SPLITS} ==========")

        # Convertimos posiciones del split a IDs reales del dataset.
        train_ids = train_idx[train_pos]
        val_ids = train_idx[val_pos]

        # Si usamos W&B, agrupamos los folds dentro de un mismo proyecto.
        if wandb is not None:
            wandb.init(
                project=WANDB_PROJECT,
                name=f"{model_name}_fold_{fold}",
                config={
                    "model_name": model_name,
                    "fold": fold,
                    "seed": SEED,
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "n_splits": N_SPLITS,
                },
                reinit=True,
            )

        model, score, history = train_one_fold(
            dataset=dataset,
            train_ids=train_ids,
            val_ids=val_ids,
            model_name=model_name,
            device=device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
        )

        print(f"Fold {fold} best RMSE: {score:.4f}")

        if wandb is not None and wandb.run is not None:
            wandb.summary["best_fold_rmse"] = score
            wandb.finish()

        fold_models.append(model)
        fold_scores.append(float(score))
        fold_histories.append(history)

    return fold_models, fold_scores, fold_histories


def save_best_model(fold_models, fold_scores, dataset):
    """
    Guarda el mejor modelo individual entre los folds.

    Aunque en el notebook se usaba ensemble, para producción inicial
    guardaremos un modelo único. Es más simple para API, Docker y tests.
    """

    best_fold_idx = int(np.argmin(fold_scores))
    best_model = fold_models[best_fold_idx]
    best_score = fold_scores[best_fold_idx]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": MODEL_NAME,
        "model_state_dict": best_model.state_dict(),
        "num_features": dataset.num_features,
        "best_fold": best_fold_idx + 1,
        "best_rmse": best_score,
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
        },
    }

    torch.save(checkpoint, MODEL_PATH)

    print(f"\nBest model saved at: {MODEL_PATH}")
    print(f"Best fold: {best_fold_idx + 1}")
    print(f"Best RMSE: {best_score:.4f}")


def save_metrics(fold_scores, fold_histories):
    """
    Guarda métricas en JSON para trazabilidad local.
    Esto complementa W&B y deja evidencia dentro del repo local.
    """

    metrics_path = MODELS_DIR / "training_metrics.json"

    metrics = {
        "model_name": MODEL_NAME,
        "fold_scores": fold_scores,
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "histories": fold_histories,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Training metrics saved at: {metrics_path}")


def main():
    """
    Punto de entrada principal del entrenamiento.
    Esto permite ejecutar:

    python src/train.py
    """

    seed_everything(SEED)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = TournamentDataset(root=str(DATA_DIR))

    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of node features: {dataset.num_features}")
    print(f"Training graphs: {len(dataset.train_idx)}")
    print(f"Test graphs: {len(dataset.test_idx)}")

    fold_models, fold_scores, fold_histories = run_kfold_training(
        dataset=dataset,
        model_name=MODEL_NAME,
        device=device,
    )

    print("\n========== Cross-validation summary ==========")
    print(f"Scores: {fold_scores}")
    print(f"Mean RMSE: {np.mean(fold_scores):.4f}")
    print(f"Std RMSE: {np.std(fold_scores):.4f}")

    save_best_model(
        fold_models=fold_models,
        fold_scores=fold_scores,
        dataset=dataset,
    )

    save_metrics(
        fold_scores=fold_scores,
        fold_histories=fold_histories,
    )


if __name__ == "__main__":
    main()