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
