from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from utils import TransactionInput, transform_input

# Inicializamos la app
app = FastAPI(title="PaySim Fraud Detection API", version="1.0")

# Configuramos CORS para permitir peticiones desde tu frontend (ej. localhost:5173 o 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Variable global para guardar el modelo
modelo_fraude = None

# Evento que se ejecuta al iniciar el servidor
@app.on_event("startup")
def load_model():
    global modelo_fraude
    model_path = "../models/fraud_model.pkl"
    
    # Verificamos si el archivo del modelo existe antes de intentar cargarlo
    if os.path.exists(model_path):
        modelo_fraude = joblib.load(model_path)
        print("✅ Modelo cargado correctamente.")
    else:
        print(f"⚠️ Advertencia: No se encontró el modelo en {model_path}. Asegúrate de exportarlo desde el Notebook.")

# Endpoint principal para la predicción
@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    # Si el modelo no cargó, lanzamos un error 500 (Internal Server Error)
    if modelo_fraude is None:
        raise HTTPException(status_code=500, detail="El modelo de predicción no está disponible.")
    
    try:
        # 1. Transformamos los datos recibidos usando nuestra función de utils.py
        input_df = transform_input(transaction)
        
        # 2. Hacemos la predicción (0 = Normal, 1 = Fraude)
        prediction = modelo_fraude.predict(input_df)
        
        # 3. (Opcional pero recomendado) Obtenemos la probabilidad para mostrar estadísticas en el frontend
        probabilities = modelo_fraude.predict_proba(input_df)
        fraud_probability = probabilities[0][1] # Probabilidad de pertenecer a la clase 1 (Fraude)
        
        # 4. Retornamos la respuesta en formato JSON (FastAPI lo convierte automáticamente)
        return {
            "is_fraud": bool(prediction[0] == 1),
            "fraud_probability": round(float(fraud_probability) * 100, 2), # En porcentaje
            "message": "Transacción analizada con éxito."
        }
        
    except Exception as e:
        # Si algo falla en la transformación o predicción, devolvemos un error claro
        raise HTTPException(status_code=400, detail=f"Error procesando la predicción: {str(e)}")