import streamlit as st
import requests
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:8000/predict"

# ── Page config (must be first Streamlit command) ──
st.set_page_config(
    page_title="PreSIADA | Fraud Detection",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e8f0fe;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #d2e3fc;
    }

    /* Submit button */
    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1557b0;
        color: #ffffff;
        border: none;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #1a73e8;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.5rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.title("PreSIADA")
    st.caption("Sistema de Detección de Fraude")
    st.markdown("---")
    page = st.radio("Navegación", ["Dashboard", "Análisis de Transacciones"])

# ═════════════════════════════════════════════════
#  PAGE 1: Dashboard
# ═════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Estadísticas y Descripción del Modelo")
    st.markdown(
        "Clasificador **Random Forest** entrenado sobre transacciones financieras "
        "del dataset PaySim. El modelo utiliza 100 estimadores optimizados para "
        "detectar patrones de fraude con alta precisión."
    )

    st.markdown("---")

    # Key metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", "99.7%", help="Porcentaje de predicciones correctas sobre el total")
    with c2:
        st.metric("Precision", "98.5%", help="Proporción de fraudes detectados que realmente son fraude")
    with c3:
        st.metric("Recall", "82.3%", help="Proporción de fraudes reales que el modelo identificó")

    st.markdown("---")

    # Performance chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metricas = ["Accuracy", "Precision", "Recall"]
    valores = [99.7, 98.5, 82.3]
    colores = ["#1a73e8", "#4285f4", "#669df6"]

    bars = ax.barh(metricas, valores, color=colores, edgecolor="white", linewidth=2, height=0.5)
    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val}%",
            va="center",
            fontweight="bold",
            fontsize=12,
            color="#174ea6",
        )

    ax.set_xlim(0, 112)
    ax.set_xlabel("Porcentaje (%)", fontsize=11, color="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", colors="#555555")
    ax.set_facecolor("#fafbfc")
    fig.patch.set_facecolor("#f5f7fa")

    st.pyplot(fig)

    st.caption(
        "Métricas obtenidas durante la evaluación del modelo con validación cruzada. "
        "El rendimiento puede variar ligeramente según la muestra de datos utilizada."
    )

# ═════════════════════════════════════════════════
#  PAGE 2: Análisis de Transacciones
# ═════════════════════════════════════════════════
else:
    st.title("Análisis de Transacciones")
    st.markdown("Complete los datos de la transacción y presione **Analizar** para obtener una predicción.")

    with st.form("prediction_form"):
        st.markdown("### Datos de la Transacción")
        col_a, col_b = st.columns(2)

        with col_a:
            step = st.number_input("Step", min_value=1, step=1, value=1, help="Número de paso de la transacción")
            amount = st.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", value=0.0)
            oldbalanceOrg = st.number_input("oldbalanceOrg", min_value=0.0, step=100.0, format="%.2f", value=0.0)
            newbalanceOrig = st.number_input("newbalanceOrig", min_value=0.0, step=100.0, format="%.2f", value=0.0)

        with col_b:
            oldbalanceDest = st.number_input("oldbalanceDest", min_value=0.0, step=100.0, format="%.2f", value=0.0)
            newbalanceDest = st.number_input("newbalanceDest", min_value=0.0, step=100.0, format="%.2f", value=0.0)
            transaction_type = st.selectbox(
                "Tipo de Transacción",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            )
            isFlaggedFraud = st.checkbox("isFlaggedFraud", help="Indica si la transacción fue marcada por el sistema")

        submitted = st.form_submit_button("Analizar Transacción")

    if submitted:
        # One-hot encode transaction type
        type_map = {t: 0 for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]}
        type_map[transaction_type] = 1

        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalanceOrg),
            "newbalanceOrig": float(newbalanceOrig),
            "oldbalanceDest": float(oldbalanceDest),
            "newbalanceDest": float(newbalanceDest),
            "type_CASH_IN": type_map["CASH_IN"],
            "type_CASH_OUT": type_map["CASH_OUT"],
            "type_DEBIT": type_map["DEBIT"],
            "type_PAYMENT": type_map["PAYMENT"],
            "type_TRANSFER": type_map["TRANSFER"],
            "isFlaggedFraud": int(isFlaggedFraud),
        }

        try:
            with st.spinner("Analizando la transacción..."):
                response = requests.post(BACKEND_URL, json=payload, timeout=30)

            if response.status_code == 422:
                st.error(
                    "El backend rechazó la solicitud (error de validación). "
                    "Verifique que el servidor esté actualizado para aceptar los campos enviados."
                )
            elif not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Error del servidor ({response.status_code}): {detail}")
            else:
                result = response.json()
                is_fraud = result.get("is_fraud", False)
                prob = result.get("fraud_probability", 0)
                nivel = result.get("nivel_alerta", "N/A")

                # Visual result card
                if is_fraud:
                    bg, bd, status = "#fce8e6", "#d93025", "FRAUDE DETECTADO"
                elif prob >= 70:
                    bg, bd, status = "#fce8e6", "#d93025", "ALTO RIESGO DE FRAUDE"
                elif prob >= 30:
                    bg, bd, status = "#fef7e0", "#f9ab00", "TRANSACCION SOSPECHOSA"
                else:
                    bg, bd, status = "#e6f4ea", "#1e8e3e", "TRANSACCION NORMAL"

                st.markdown(
                    f"""
                    <div style="
                        background:{bg};
                        border:2px solid {bd};
                        border-radius:14px;
                        padding:28px 32px;
                        margin:8px 0;
                    ">
                        <p style="
                            color:{bd};
                            font-size:0.95rem;
                            font-weight:600;
                            margin:0 0 8px 0;
                            text-transform:uppercase;
                            letter-spacing:0.5px;
                        ">
                            {status}
                        </p>
                        <p style="
                            font-size:2.8rem;
                            font-weight:700;
                            margin:8px 0;
                            color:#1a1a1a;
                        ">
                            {prob:.2f}%
                        </p>
                        <p style="
                            font-size:1rem;
                            color:#444444;
                            margin:0;
                        ">
                            Nivel de alerta: <strong>{nivel}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Extra context
                with st.expander("Detalles de la predicción"):
                    st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con el backend. Asegúrese de que el servidor esté corriendo en `localhost:8000`.")
        except requests.exceptions.Timeout:
            st.error("La solicitud excedió el tiempo de espera. Intente nuevamente.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión: {e}")
        except Exception as e:
            st.error(f"Error inesperado: {e}")
