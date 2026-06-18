import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/predict"
TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

st.set_page_config(
    page_title="Trabajo Práctico Final de Python",
    layout="centered",
)

st.title("Modelo de Detección de Fraude Financiero")
st.write("Ingrese los valores de la transacción para obtener una predicción.")

with st.form("fraud_form"):
    step = st.number_input(
        "step",
        min_value=0,
        step=1,
        format="%d",
    )

    type_ = st.selectbox(
        "type",
        options=TRANSACTION_TYPES,
    )

    amount = st.number_input(
        "amount",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    oldbalanceOrg = st.number_input(
        "oldbalanceOrg",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    newbalanceOrig = st.number_input(
        "newbalanceOrig",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    oldbalanceDest = st.number_input(
        "oldbalanceDest",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    newbalanceDest = st.number_input(
        "newbalanceDest",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    submitted = st.form_submit_button("Analizar transacción")

if submitted:
    payload = {
        "step": int(step),
        "type": type_,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
    }

    try:
        with st.spinner("Analizando la transacción..."):
            response = requests.post(BACKEND_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

        message = result.get("message", "Transacción analizada con éxito.")
        fraud_probability = result.get("fraud_probability")
        alert_level = result.get("nivel_alerta")

        st.success(message)

        col2, col3 = st.columns(3)
        #with col1:
            #st.metric(
                #"Resultado",
                #"Fraude" if result.get("is_fraud") else "Normal",
            #)
        with col2:
            st.metric(
                "Probabilidad de fraude",
                f"{fraud_probability:.2f}%" if fraud_probability is not None else "",
            )
        with col3:
            st.metric(
                "Nivel de alerta",
                alert_level if alert_level is not None else "",
            )

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la respuesta: {e}")