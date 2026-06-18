# AGENTS.md — tp_final_py

## Project Overview

Fraud detection ML app (PaySim dataset). FastAPI backend + Streamlit frontend + scikit-learn model. Educational/training project ("Trabajo Práctico Final").

## Tech Stack

- **Python 3.13**, managed with **uv** (not pip/poetry)
- **FastAPI** backend (`backend/main.py`)
- **Streamlit** frontend (`frontend/app.py`)
- **scikit-learn** model trained in Jupyter notebook (`model/model_main.ipynb`)
- Data exploration notebook: `data/main_doc.ipynb`

## Commands

```bash
# Install dependencies
uv sync

# Run backend (port 8000)
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend (Streamlit, default port 8501)
uv run streamlit run frontend/app.py

# Run root main.py (placeholder, not the app entrypoint)
uv run python main.py
```

There are no tests, lint, typecheck, or formatter configs defined yet.

## Architecture

```
main.py                 # Placeholder entrypoint (not used by app)
backend/
  main.py               # FastAPI app, CORS, lifespan, Pydantic schema, /metrics, /predict
frontend/
  app.py                # Streamlit UI, POSTs to http://localhost:8000/predict
model/
  model_main.ipynb      # Model training notebook — exports .pkl
data/
  main_doc.ipynb        # Data exploration notebook
```

## Key Details

- **Model path**: backend resolves `../models/fraud_model.pkl` relative to `backend/main.py`, so the trained model must be saved to `models/fraud_model.pkl` at repo root. The notebook must create the `models/` directory.
- **Input schema** (`TransactionInput`): `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`, plus one-hot booleans for transaction type: `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`. Exactly one type flag must be `true`.
- **Feature order** matters: the backend builds the DataFrame in a fixed `FEATURE_ORDER` before calling `model.predict()`. If the model was trained with a different column order, update `FEATURE_ORDER` in `backend/main.py`.
- **CORS** allows all origins (`["*"]`) for development.
- **Endpoints**: `GET /metrics` returns simulated accuracy/precision/recall/f1_score; `POST /predict` returns `{is_fraud, probability, message}`.
- **Frontend expects backend at** `http://localhost:8000/predict`.
- Transaction types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`.
- `data.csv` and `dataset.zip` are gitignored (large data files).
