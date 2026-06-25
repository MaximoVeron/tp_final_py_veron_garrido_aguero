import joblib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.pkl"

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Fraud Detection API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    type: str = Field(..., pattern="^(CASH_OUT|TRANSFER)$")
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


def _build_features(tx: TransactionInput) -> pd.DataFrame:
    error_balance_orig = tx.newbalanceOrig + tx.amount - tx.oldbalanceOrg
    error_balance_dest = tx.oldbalanceDest + tx.amount - tx.newbalanceDest
    vacia_cuenta = 1 if tx.amount == tx.oldbalanceOrg else 0

    return pd.DataFrame([{
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "error_balance_orig": error_balance_orig,
        "error_balance_dest": error_balance_dest,
        "vacia_cuenta": vacia_cuenta,
        "type_TRANSFER": 1 if tx.type == "TRANSFER" else 0,
    }])


def _nivel_alerta(prob: float) -> str:
    if prob < 0.35:
        return "Improbable"
    elif prob <= 0.75:
        return "Sospechoso"
    else:
        return "Fraude Altamente Probable"


@app.post("/predict")
def predict(transaction: TransactionInput):
    try:
        df = _build_features(transaction)
        prob = float(_model.predict_proba(df)[0][1])
        nivel = _nivel_alerta(prob)
        return {
            "fraud_probability": round(prob * 100, 2),
            "nivel_alerta": nivel,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en prediccion: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
