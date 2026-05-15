<div align="center">

<img src="https://masterdeeplearning.etsisi.upm.es/wp-content/uploads/2024/02/wallpaper-full-front-copia-4.png" width="100%"/>

<img src="https://masterdeeplearning.etsisi.upm.es/wp-content/uploads/2024/09/LOGOTIPO-leyenda-color-JPG-p-1024x475.png" width="400px"/>


# GNN Lipophilicity — MLOps Project

| | |
|---|---|
| **Autores** | David Nicolás Tamajón Rivas <br> Aliss Maria Bejerano Kindelan |
| **Modelo** | GraphSAGE |
| **Dataset** | MoleculeNet — Lipophilicity |

[![Python](https://img.shields.io/badge/Python-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)

</div>

<div align="left">
    
## Descripción del proyecto

Este proyecto corresponde a la entrega final de la asignatura de MLOps del Máster en Deep Learning.

El proyecto original consistía en predecir la lipofilicidad molecular utilizando Graph Neural Networks. En esta versión, el objetivo ha sido transformar el notebook experimental original en un proyecto de Machine Learning más cercano a producción, aplicando buenas prácticas de MLOps.

El proyecto incluye:

- Código modular en Python extraído del notebook original.
- Pipeline de entrenamiento para un modelo Graph Neural Network.
- Seguimiento de experimentos con Weights & Biases.
- Guardado del modelo entrenado.
- Utilidades de inferencia local.
- API de inferencia con FastAPI.
- Tests automatizados con pytest.
- Contenedorización con Docker.
- Documentación para ejecución local.

## Problema

La tarea consiste en predecir la lipofilicidad molecular a partir de representaciones de moléculas como grafos.

Cada molécula se representa como un grafo:

- Los nodos representan átomos o entidades moleculares.
- Las aristas representan relaciones o enlaces entre nodos.
- Las características de los nodos se utilizan como entrada del modelo.
- La variable objetivo es la lipofilicidad.

Se trata de un problema de regresión, ya que el modelo devuelve una predicción numérica continua.

## Modelo

El modelo seleccionado como candidato final es una Graph Neural Network basada en GraphSAGE con agregación mediante `global_mean_pool`.

Este modelo fue seleccionado a partir de la fase experimental original por ofrecer un rendimiento competitivo en validación y una arquitectura adecuada para ser integrada en un flujo MLOps.

La métrica principal utilizada es:

```text
RMSE - Root Mean Squared Error
```

## Estructura del repositorio

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── pytest.ini
├── .dockerignore
├── .gitignore
│
├── data/
│   └── ...
│
├── models/
│   └── sage_mean_model.pt
│
├── notebooks/
│   └── PRACTICA_GNN_Lipophilicity.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── api_inferencia.py
│   └── utils.py
│
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    ├── test_predict.py
    └── test_api.py
```

## Configuración del entorno local

Crear y activar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entrenamiento del modelo

Para entrenar el modelo localmente:

```bash
python -m src.train
```

El script de entrenamiento realiza los siguientes pasos:

- Fija la semilla para mejorar la reproducibilidad.
- Carga y procesa el dataset de grafos moleculares.
- Entrena el modelo GraphSAGE usando validación K-Fold.
- Registra métricas en Weights & Biases.
- Guarda el mejor checkpoint del modelo en la carpeta models/.
- Guarda métricas locales de entrenamiento en formato JSON.

Archivos esperados después del entrenamiento:
- models/sage_mean_model.pt
- models/training_metrics.json

## Weights & Biases

El proyecto utiliza Weights & Biases para el seguimiento de experimentos.

Se registran, entre otros, los siguientes elementos:

- Nombre del modelo.
- Número de fold.
-Semilla aleatoria.
- Batch size.
- Learning rate.
- Weight decay.
- Pérdida de entrenamiento.
- RMSE de validación.
- Mejor RMSE de validación.

Enlace al proyecto o report de W&B: 
```link
https://wandb.ai/ntamajon-universidad-polit-cnica-de-madrid/gnn-lipophilicity-mlops/reports/MLOps-GNN-Lipophilicity-Experiment-Report--VmlldzoxNjg4NDMwNw?accessToken=agpkl2vfcs2jlcx18vgplzenflnf6vlbl9vgdp8auorcovq7pasaci8ezky1wot1
```

## Inferencia local

Para ejecutar una inferencia local usando un grafo del dataset procesado:

```bash
python -m src.predict
```

Este comando carga el checkpoint del modelo entrenado y predice la lipofilicidad de un ejemplo del dataset.

## API de inferencia

El proyecto incluye una API con FastAPI para realizar inferencia local.

Para levantar la API:

```bash
python -m uvicorn src.api_inferencia:app --host 127.0.0.1 --port 8000
```
Despues, abre el navegador y accede a la documentación interactiva de la API en:
http://127.0.0.1:8000/docs

Endpoints disponibles:
GET /
GET /health
POST /predict-by-index

Ejemplo de petición:

```json
{
  "index": 0
}
```

Ejemplo de respuesta:

```JSON
{
  "index": 0,
  "prediction": 1.2345,
  "target": "lipophilicity",
  "model_name": "SAGE_mean",
  "best_rmse": 0.72
}
```

El endpoint predice la lipofilicidad de un grafo molecular ya disponible en el dataset procesado. Este endpoint permite validar el flujo completo de inferencia MLOps: carga del modelo, carga del dato, predicción y respuesta JSON.

## Tests

Para ejecutar los tests:
```bash
python -m pytest
```

Los tests validan:

- Carga del dataset.
- Creación del modelo.
- Existencia del checkpoint entrenado.
- Inferencia local.
- Funcionamiento de los endpoints de la API.
- Manejo de errores de la API.


## Docker

Construir la imagen Docker:
```bash
docker build -t gnn-lipophilicity-api .
```

Correr el contenedor:
```bash
docker run -p 8000:8000 gnn-lipophilicity-api
```

Después abrir:
http://127.0.0.1:8000/docs

El contenedor Docker arranca automáticamente la API de inferencia con FastAPI.

## Despliegue cloud

El proyecto incluye una API de inferencia local y contenedorización con Docker.

No se ha realizado despliegue en cloud porque, según la aclaración del profesor, esta parte no es obligatoria para la entrega. El foco principal del trabajo está en la reproducibilidad, modularización, seguimiento de experimentos, testing, contenedorización e inferencia local.


## Repositorio GitHub

Repositorio GitHub: 
```link
https://github.com/ntamajon/gnn-lipophilicity-mlops
```
</div>
