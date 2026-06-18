import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Trabajo Práctico Final de Python",
    layout="centered",
)

st.title("Modelo de Detección de Fraude Financiero")
st.write("Ingrese los valores de la transacción para obtener una predicción.")

with st.form("fraud_form"):
    step = st.number_input(
        "Paso (step)",
        min_value=1,
        step=1,
        help="Número de paso de la transacción.",
    )

    transaction_type = st.selectbox(
        "Tipo de transacción (type)",
        options=[
            "CASH_OUT",
            "PAYMENT",
            "CASH_IN",
            "TRANSFER",
            "DEBIT",
        ],
    )

    amount = st.number_input(
        "Monto (amount)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    oldbalance_org = st.number_input(
        "Saldo inicial origen (oldbalanceOrg)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    newbalance_orig = st.number_input(
        "Saldo final origen (newbalanceOrig)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    oldbalance_dest = st.number_input(
        "Saldo inicial destino (oldbalanceDest)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    newbalance_dest = st.number_input(
        "Saldo final destino (newbalanceDest)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
    )

    submitted = st.form_submit_button("Analizar transacción")

if submitted:
    payload = {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "newbalanceOrig": float(newbalance_orig),
        "oldbalanceDest": float(oldbalance_dest),
        "newbalanceDest": float(newbalance_dest),
    }

    try:
        with st.spinner("Analizando la transacción..."):
            response = requests.post(BACKEND_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

        st.success(result.get("message", "Predicción realizada correctamente."))

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Resultado",
                "Fraude" if result.get("is_fraud") else "Normal",
            )
        with col2:
            st.metric(
                "Probabilidad de fraude",
                f"{result.get('fraud_probability', 0):.2f}%",
            )

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la respuesta: {e}")