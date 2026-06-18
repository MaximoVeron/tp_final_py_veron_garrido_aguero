# AGENTS.md — tp_final_py

## Project Overview

Fraud detection ML app (PaySim dataset). FastAPI backend + Streamlit frontend + scikit-learn model. Educational/training project ("Trabajo Práctico Final").

## Tech Stack

- **Python 3.13**, managed with **uv** (not pip/poetry)
- **FastAPI** backend (`backend/main.py`, `backend/utils.py`)
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
  main.py               # FastAPI app, loads model at startup, serves POST /predict
  utils.py              # Pydantic schema (TransactionInput), transform_input()
frontend/
  app.py                # Streamlit UI, POSTs to http://localhost:8000/predict
model/
  model_main.ipynb      # Model training notebook — exports .pkl
data/
  main_doc.ipynb        # Data exploration notebook
```

## Key Details

- **Model path**: backend loads `../models/fraud_model.pkl` (relative to `backend/`), so the trained model must be saved to `models/fraud_model.pkl` at repo root. This directory does not exist yet — the notebook must create it.
- **Model must be an sklearn Pipeline** that handles the `type` categorical feature internally (OneHotEncoder/LabelEncoder). The `transform_input()` in `utils.py` passes raw values through; any encoding must be inside the pipeline.
- **CORS** allows `localhost:5173`, `localhost:3000`, `127.0.0.1:5173`.
- **Frontend expects backend at** `http://localhost:8000/predict`.
- Transaction types: `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`.
- `data.csv` and `dataset.zip` are gitignored (large data files).
