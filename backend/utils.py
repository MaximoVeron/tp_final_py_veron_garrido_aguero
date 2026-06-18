# utils.py
from pydantic import BaseModel
import pandas as pd

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

class TransactionInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

def transform_input(data: TransactionInput) -> pd.DataFrame:
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Replicamos exactamente el get_dummies del notebook
    df = pd.get_dummies(df, columns=["type"])

    # Forzamos que existan TODAS las columnas que el modelo entrenó,
    # rellenando con 0 las que no aparezcan en esta transacción
    for t in TRANSACTION_TYPES:
        col = f"type_{t}"
        if col not in df.columns:
            df[col] = 0

    # Garantizamos el orden de columnas exacto que vio el modelo
    expected_cols = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER"
    ]
    df = df[expected_cols]

    return df