# PreSIADA - Sistema de Detección de Fraude Financiero

Sistema completo de detección de fraudes en transacciones financieras usando Machine Learning, FastAPI y Streamlit.

## Descripción

Este proyecto implementa un clasificador **Random Forest** entrenado sobre el dataset PaySim para detectar transacciones fraudulentas de tipo `CASH_OUT` y `TRANSFER`.

### Características

- **Backend (FastAPI):** API REST que carga el modelo en memoria y realiza predicciones en tiempo real
- **Frontend (Streamlit):** Interfaz web con dashboard de métricas y simulador de riesgo
- **Feature Engineering:** Cálculo automático de variables sintéticas para mejorar la detección
- **Alta precisión:** 99.7% de accuracy, 98.0% de precision, 99.6% de recall

## Estructura del Proyecto

```
.
├── api.py                      # Backend FastAPI (nuevo)
├── frontend/
│   └── app.py                  # Frontend Streamlit (actualizado)
├── models/
│   └── fraud_model.pkl         # Modelo Random Forest entrenado
├── train_fraud_model.py        # Script de entrenamiento
├── data.csv                    # Dataset PaySim (gitignored)
├── pyproject.toml              # Dependencias del proyecto
└── README.md                   # Este archivo
```

## Instalación

### Requisitos

- Python 3.13+
- uv (gestor de paquetes)

### Pasos

1. **Instalar dependencias:**

```bash
uv sync
```

2. **Entrenar el modelo (si no existe):**

```bash
uv run python train_fraud_model.py
```

Esto generará `models/fraud_model.pkl`.

## Uso

### 1. Iniciar el Backend (FastAPI)

```bash
uv run uvicorn api:app --reload --port 8000
```

El servidor estará disponible en `http://localhost:8000`.

**Endpoints:**
- `GET /health` - Verificar estado del servidor
- `POST /predict` - Realizar predicción de fraude

**Ejemplo de request:**

```json
{
  "type": "TRANSFER",
  "amount": 50000.0,
  "oldbalanceOrg": 100000.0,
  "newbalanceOrig": 50000.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 50000.0
}
```

**Respuesta:**

```json
{
  "fraud_probability": 0.0,
  "nivel_alerta": "Improbable"
}
```

### 2. Iniciar el Frontend (Streamlit)

En otra terminal:

```bash
uv run streamlit run frontend/app.py
```

La interfaz estará disponible en `http://localhost:8501`.

### Páginas disponibles:

#### Dashboard del Modelo
- Métricas de rendimiento (Accuracy, Precision, Recall)
- Matriz de confusión visual
- Estadísticas de detección

#### Simulador de Riesgo
- Formulario para ingresar datos de transacciones
- Predicción en tiempo real
- Visualización de nivel de riesgo:
  - ✅ **Improbable** (< 35%)
  - ⚠️ **Sospechoso** (35% - 75%)
  - 🚨 **Fraude Altamente Probable** (> 75%)

## Feature Engineering

El backend calcula automáticamente las siguientes variables sintéticas:

- `error_balance_orig = newbalanceOrig + amount - oldbalanceOrg`
- `error_balance_dest = oldbalanceDest + amount - newbalanceDest`
- `vacia_cuenta = 1` si `amount == oldbalanceOrg`, sino `0`

Estas features ayudan al modelo a detectar inconsistencias en los saldos y patrones sospechosos.

## Métricas del Modelo

| Métrica   | Valor  | Descripción                                      |
|-----------|--------|--------------------------------------------------|
| Accuracy  | 99.7%  | Porcentaje de predicciones correctas             |
| Precision | 98.0%  | Fraudes detectados que son reales                |
| Recall    | 99.6%  | Fraudes reales que fueron detectados             |
| F1-Score  | 98.8%  | Balance entre precision y recall                 |

### Matriz de Confusión

|                  | Pred: Normal | Pred: Fraude |
|------------------|--------------|--------------|
| **Real: Normal** | 128,437 (TN) | 30 (FP)      |
| **Real: Fraude** | 6 (FN)       | 1,637 (TP)   |

## Tecnologías

- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit, Matplotlib
- **ML:** scikit-learn, pandas, numpy, joblib
- **Dataset:** PaySim (simulación de transacciones financieras)

## Notas

- El modelo solo acepta transacciones de tipo `CASH_OUT` y `TRANSFER`
- El archivo `data.csv` está en `.gitignore` por su tamaño
- Asegúrate de que el backend esté corriendo antes de usar el frontend
- El modelo se carga en memoria al iniciar el servidor (lifespan de FastAPI)

## Licencia

Proyecto académico - TP Final Python
