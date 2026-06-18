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
from pydantic import BaseModel, Field, field_validator, model_validator

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
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
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

    El frontend debe convertir la selección del tipo de transacción en las
    variables booleanas one-hot definidas a continuación.
    """

    step: int = Field(..., ge=1, le=744, description="Paso temporal de la transacción (1–744).")
    amount: float = Field(..., ge=0, description="Monto de la transacción.")
    oldbalanceOrg: float = Field(..., ge=0, description="Saldo origen antes de la transacción.")
    newbalanceOrig: float = Field(..., ge=0, description="Saldo origen después de la transacción.")
    oldbalanceDest: float = Field(..., ge=0, description="Saldo destino antes de la transacción.")
    newbalanceDest: float = Field(..., ge=0, description="Saldo destino después de la transacción.")
    isFlaggedFraud: int = Field(..., ge=0, le=1, description="Flag previo de fraude (0 o 1).")

    type_CASH_IN: bool = Field(..., description="Indica si el tipo es CASH_IN.")
    type_CASH_OUT: bool = Field(..., description="Indica si el tipo es CASH_OUT.")
    type_DEBIT: bool = Field(..., description="Indica si el tipo es DEBIT.")
    type_PAYMENT: bool = Field(..., description="Indica si el tipo es PAYMENT.")
    type_TRANSFER: bool = Field(..., description="Indica si el tipo es TRANSFER.")

    @field_validator("isFlaggedFraud", mode="before")
    @classmethod
    def coerce_flag(cls, v):
        """Permite recibir el flag como booleano y convertirlo a entero."""
        if isinstance(v, bool):
            return int(v)
        return v

    @model_validator(mode="after")
    def check_exactly_one_type(self):
        """Garantiza que exactamente una categoría de transacción esté activa."""
        type_flags = [
            self.type_CASH_IN,
            self.type_CASH_OUT,
            self.type_DEBIT,
            self.type_PAYMENT,
            self.type_TRANSFER,
        ]
        if sum(type_flags) != 1:
            raise ValueError(
                "Debe seleccionarse exactamente un tipo de transacción (one-hot encoding)."
            )
        return self


class PredictionResponse(BaseModel):
    """Esquema de salida para el endpoint /predict."""

    is_fraud: bool
    probability: float
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
    row = {feature: data[feature] for feature in FEATURE_ORDER}
    return pd.DataFrame([row])


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
        is_fraud = bool(prediction[0] == 1)

        # 3. Probabilidad (si el modelo la soporta)
        probability = 0.0
        if hasattr(modelo_fraude, "predict_proba"):
            proba = modelo_fraude.predict_proba(input_df)
            # Clase 1 = fraude
            probability = float(proba[0][1]) * 100

        # 4. Mensaje legible
        message = "Alerta de Fraude" if is_fraud else "Transacción Segura"

        return PredictionResponse(
            is_fraud=is_fraud,
            probability=round(probability, 2),
            message=message,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error procesando la predicción: {str(exc)}",
        )
