from pydantic import BaseModel
import pandas as pd


# 1. Definimos qué datos esperamos recibir del frontend y sus tipos.
# Pydantic validará automáticamente que no falten campos o envíen texto en lugar de números.
class TransactionInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# 2. Función para transformar este JSON en un DataFrame de Pandas
def transform_input(data: TransactionInput) -> pd.DataFrame:
    # Convertimos los datos validados a un diccionario y luego a un DataFrame de 1 fila
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    
    # NOTA PARA TU EQUIPO: 
    # Si en su Jupyter Notebook hicieron transformaciones (ej. convertir "type" a números 
    # usando OneHotEncoder o LabelEncoder), DEBEN aplicar esa misma lógica aquí antes de retornar el df.
    # Lo ideal es que el archivo .pkl sea un "Pipeline" de sklearn que ya incluya la transformación.
    
    return df