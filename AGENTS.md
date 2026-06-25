<<<<<<< HEAD
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
=======
# AGENTS.md

Compact guidance for OpenCode sessions working on this repo.

## Project shape

- Small Python final project: FastAPI backend + Streamlit frontend + scikit-learn training notebook.
- Managed with `uv`: Python 3.13, `.venv` present, `pyproject.toml` + `uv.lock`.
- No tests, no CI, no lint/typecheck config. Treat it as a simple student/experiment repo.

## Entrypoints

- **Backend:** `backend/main.py` (FastAPI). Serve it from the `backend/` directory because it uses `from utils import ...`.
  ```powershell
  uv run --directory backend uvicorn main:app --reload --port 8000
  ```
- **Frontend:** `frontend/app.py` (Streamlit).
  ```powershell
  uv run streamlit run frontend/app.py
  ```
- **Training:** `model/model_main.ipynb` reads `../data.csv` and trains a RandomForest fraud classifier.

## Known gotchas

- **Model artifact path mismatch.** The backend loads `../models/fraud_model.pkl`, but the repo has a `model/` directory (singular) and no `.pkl` file is committed. The notebook currently does not export a model. To run predictions you must either create `models/fraud_model.pkl` from the notebook or update the path in `backend/main.py`.
- **Streamlit is not declared in `pyproject.toml`.** It is installed in the local `.venv`, but it is missing from project dependencies. Add it if you recreate the environment.
- **Backend has a runtime bug in `/predict`.** `backend/main.py` defines `nivel_alerta(...)` but then calls an undefined `mapear_nivel_alerta(...)` inside that function, so successful predictions will raise a `NameError`.
- **Startup uses deprecated FastAPI API.** `@app.on_event("startup")` is deprecated; expect a warning.
- **`data.csv` is gitignored.** It lives at repo root and is read by the notebook as `../data.csv`.

## Workflow notes

- No test command exists. Verification is manual: run backend, run frontend, or execute the notebook.
- The `data/` directory only contains a mostly empty notebook (`main_doc.ipynb`); the real dataset is `data.csv` at the root.
- `README.md` is empty; this file and the code are the only sources of truth.
>>>>>>> 3037c8362e0d4b7ae5375efc5caecdcd6d85c43c
