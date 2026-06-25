"""
FraudShield — Backend API
=========================
API REST construida con FastAPI para servir un modelo de detección de fraude
transaccional entrenado con scikit-learn y exportado como archivo .pkl.

Ejecución local (desde la raíz del proyecto):
    uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Ejecución local (desde el directorio backend/):
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudshield")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH = (Path(__file__).parent / "../models/fraud_model.pkl").resolve()

# Orden EXACTO de las características con el que fue entrenado el modelo.
# Este orden se respeta al construir el DataFrame que se envía a scikit-learn.
FEATURE_ORDER = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "error_balance_orig",
    "error_balance_dest",
    "vacia_cuenta",
    "type_TRANSFER",
]

modelo_fraude: Any = None


# ═══════════════════════════════════════════════════════════════════════════
# CICLO DE VIDA — Carga del modelo
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo de scikit-learn al iniciar la aplicación."""
    global modelo_fraude

    try:
        if MODEL_PATH.exists():
            modelo_fraude = joblib.load(MODEL_PATH)
            logger.info(f"Modelo cargado correctamente desde: {MODEL_PATH.resolve()}")
        else:
            logger.warning(
                f"No se encontro el modelo en {MODEL_PATH.resolve()}. "
                "Asegurate de exportarlo desde el notebook de entrenamiento."
            )
    except Exception as exc:
        logger.error(f"Error al cargar el modelo: {exc}")
        modelo_fraude = None

    yield

    # Limpieza al cerrar (opcional)
    modelo_fraude = None


# ═══════════════════════════════════════════════════════════════════════════
# APLICACIÓN FASTAPI
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="FraudShield API",
    description="API para detección de fraude transaccional basada en scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permitimos todos los orígenes para facilitar el consumo desde Streamlit
# o cualquier frontend durante desarrollo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ═══════════════════════════════════════════════════════════════════════════

class TransactionInput(BaseModel):
    """Esquema de entrada para una transacción a evaluar.

    El frontend envía los datos básicos y el backend calcula las features
    derivadas (error_balance_orig, error_balance_dest, vacia_cuenta).
    """

    amount: float = Field(..., ge=0, description="Monto de la transacción.")
    oldbalanceOrg: float = Field(..., ge=0, description="Saldo origen antes de la transacción.")
    newbalanceOrig: float = Field(..., ge=0, description="Saldo origen después de la transacción.")
    oldbalanceDest: float = Field(..., ge=0, description="Saldo destino antes de la transacción.")
    newbalanceDest: float = Field(..., ge=0, description="Saldo destino después de la transacción.")
    type_TRANSFER: bool = Field(..., description="True si el tipo es TRANSFER, False si es CASH_OUT.")


class PredictionResponse(BaseModel):
    """Esquema de salida para el endpoint /predict."""

    is_fraud: bool
    fraud_probability: float
    nivel_alerta: str
    message: str


class MetricsResponse(BaseModel):
    """Esquema de salida para el endpoint /metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


# ═══════════════════════════════════════════════════════════════════════════
# AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_dataframe(payload: TransactionInput) -> pd.DataFrame:
    """Construye un DataFrame con el orden exacto de características del modelo."""
    data = payload.model_dump()
    row = {feature: data[feature] for feature in FEATURE_ORDER if feature in data}

    # Calcular features derivadas que el modelo espera
    row["error_balance_orig"] = data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]
    row["error_balance_dest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
    row["vacia_cuenta"] = 1 if data["amount"] == data["oldbalanceOrg"] else 0

    return pd.DataFrame([row])[FEATURE_ORDER]


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "service": "FraudShield API",
        "version": "1.0.0",
        "model_loaded": modelo_fraude is not None,
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint de healthcheck."""
    return {"status": "ok", "model_loaded": modelo_fraude is not None}


@app.get("/metrics", response_model=MetricsResponse, tags=["Modelo"])
def get_metrics():
    """Devuelve métricas simuladas de rendimiento del modelo.

    En una versión productiva estos valores pueden leerse de un archivo de
    resultados de validación cruzada o de un experiment tracker.
    """
    return MetricsResponse(
        accuracy=0.9742,
        precision=0.9417,
        recall=0.9163,
        f1_score=0.9288,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict_fraud(transaction: TransactionInput):
    """Recibe una transacción, la evalúa con el modelo y devuelve el resultado."""
    if modelo_fraude is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de predicción no está disponible. Verifique que el archivo .pkl exista.",
        )

    try:
        # 1. Preparar la entrada respetando el orden de características
        input_df = build_feature_dataframe(transaction)

        # 2. Predicción
        prediction = modelo_fraude.predict(input_df)
        
        # 3. (Opcional pero recomendado) Obtenemos la probabilidad para mostrar estadísticas en el frontend
        probabilities = modelo_fraude.predict_proba(input_df)
        fraud_probability = probabilities[0][1] # Probabilidad de pertenecer a la clase 1 (Fraude)

        def nivel_alerta(prob):
            if prob < 0.30:
                return "Improbable"
            elif prob < 0.70:
                return "Sospechoso"
            else:
                return "Muy Probable"

        # 4. Retornamos la respuesta en formato JSON (FastAPI lo convierte automáticamente)
        return {
            "is_fraud": bool(prediction[0] == 1),
            "fraud_probability": round(float(fraud_probability) * 100, 2),
            "nivel_alerta": nivel_alerta(fraud_probability),
            "message": "Transacción analizada con éxito."
        }
        
    except Exception as e:
        # Si algo falla en la transformación o predicción, devolvemos un error claro
        raise HTTPException(status_code=400, detail=f"Error procesando la predicción: {str(e)}")
